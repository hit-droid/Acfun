/* Acfun 二进制模型格式与顺序执行器。
 *
 * 文件布局 (小端):
 *   magic[6] = "ACFUN0"
 *   u32 version
 *   u32 inC, inH, inW            (batch 固定为 1)
 *   u32 num_layers
 *   逐层记录:
 *     u32 type  (1 Conv, 2 FC, 3 ReLU, 4 MaxPool, 5 Softmax, 6 BN)
 *     Conv : u32 KH,KW,Cout,stride,pad | float w[KH*KW*Cin*Cout] | float b[Cout]
 *     FC   : u32 Cout, inDim         | float w[inDim*Cout]        | float b[Cout]
 *     ReLU : (无参数)
 *     MaxP : u32 pool,stride,pad
 *     SoftM: (无参数)
 *     BN   : u32 ch | float mean[ch] var[ch] gamma[ch] beta[ch]
 *   其中 Conv 的 Cin 取上一层输出通道。
 *
 * 加载期融合优化 (对标 MNN 的图优化):
 *   - BN 折叠进紧邻其前的 Conv/FC 权重与偏置, 推理时整层 BN 消失;
 *   - 紧跟 Conv/FC 的 ReLU 融合进卷积 (act 标志), 省去额外激活扫描。
 * 二者结合使 "Conv->BN->ReLU" 退化为单次融合卷积。 */
#include "acfun.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>

struct Layer {
    int type;
    int p[8];
    const float* w;
    const float* b;
    int act;            /* 0=无激活 1=融合 ReLU (仅 Conv/FC 使用) */
};

struct AcfunModel {
    int inC, inH, inW;
    int num;
    Layer* layers;
    unsigned char* blob;
    int outSize;
    std::vector<float*> owned;   /* 融合产生的折叠权重/偏置, destroy 时释放 */
    float* buf[2];              /* ping-pong 工作缓冲, 降低逐层 malloc 开销 */
    size_t bufc[2];
    int cur;                    /* 当前数据所在的缓冲下标 */
};

static unsigned read_u32(const unsigned char* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

/* 单条原始层 (指针指向 blob); prevC/wn 在第一遍解析时算好, 供融合与折叠使用 */
struct RawLayer {
    int type;
    int p[8];
    const float* w;   /* Conv/FC: 权重; BN: mean|var|gamma|beta 连续 */
    const float* b;   /* Conv/FC: 偏置 */
    int prevC;        /* Conv 输入通道 / FC 不需要 */
    size_t wn;        /* 权重元素总数 */
};

/* 将 BN 折叠进 Conv/FC: w'[ic][c]=w[ic][c]*scale_c, b'[c]=b[c]*scale_c+shift_c */
static void fold_bn_into(Layer& F, const RawLayer& bn, int Cout, size_t wn, AcfunModel* m) {
    int ch = bn.p[0];
    const float* mean  = bn.w;
    const float* var   = bn.w + ch;
    const float* gamma = bn.w + 2 * ch;
    const float* beta  = bn.w + 3 * ch;
    size_t per_oc = wn / (size_t)Cout;       /* 每个输出通道的输入元素数 */
    float* nw = (float*)malloc(wn * sizeof(float));
    float* nb = (float*)malloc((size_t)Cout * sizeof(float));
    if (!nw || !nb) { free(nw); free(nb); return; }
    std::memcpy(nw, F.w, wn * sizeof(float));
    for (int c = 0; c < Cout; ++c) {
        float scale = gamma[c] / sqrtf(var[c] + 1e-5f);
        float shift = beta[c] - mean[c] * scale;
        for (size_t t = 0; t < per_oc; ++t) nw[t * (size_t)Cout + c] *= scale;
        nb[c] = (F.b ? F.b[c] : 0.0f) * scale + shift;
    }
    F.w = nw; F.b = nb;
    m->owned.push_back(nw); m->owned.push_back(nb);
}

/* 保证指定缓冲容量足够, 不足则扩容 (保留旧内容无意义, 直接重分配) */
static float* ensure_buf(AcfunModel* m, int which, size_t n) {
    if (m->buf[which] && m->bufc[which] >= n) return m->buf[which];
    free(m->buf[which]);
    m->buf[which] = (float*)malloc(n * sizeof(float));
    m->bufc[which] = n;
    return m->buf[which];
}

AcfunModel* acfun_model_load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return nullptr;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* blob = (unsigned char*)malloc(sz ? sz : 1);
    if (!blob) { fclose(f); return nullptr; }
    if (fread(blob, 1, sz, f) != (size_t)sz) { fclose(f); free(blob); return nullptr; }
    fclose(f);

    if (sz < 6 + 4 * 4 || memcmp(blob, "ACFUN0", 6) != 0) { free(blob); return nullptr; }
    size_t off = 6 + 4;                                /* magic + version */
    int inC = read_u32(blob + off); off += 4;
    int inH = read_u32(blob + off); off += 4;
    int inW = read_u32(blob + off); off += 4;
    int num = read_u32(blob + off); off += 4;

    AcfunModel* m = (AcfunModel*)calloc(1, sizeof(AcfunModel));
    if (!m) { free(blob); return nullptr; }
    m->inC = inC; m->inH = inH; m->inW = inW; m->num = num;
    m->blob = blob; m->buf[0] = m->buf[1] = nullptr; m->bufc[0] = m->bufc[1] = 0; m->cur = 0;

    /* ---- 第一遍: 解析原始层, 同时算好 prevC / wn ---- */
    std::vector<RawLayer> raw(num);
    {
        size_t o = off;
        int prevC = inC;
        for (int i = 0; i < num; ++i) {
            RawLayer& L = raw[i];
            std::memset(&L, 0, sizeof(L));
            L.type = read_u32(blob + o); o += 4;
            switch (L.type) {
                case 1: { /* Conv */
                    int KH = read_u32(blob + o); o += 4;
                    int KW = read_u32(blob + o); o += 4;
                    int Cout = read_u32(blob + o); o += 4;
                    int st = read_u32(blob + o); o += 4;
                    int pd = read_u32(blob + o); o += 4;
                    L.p[0]=KH; L.p[1]=KW; L.p[2]=Cout; L.p[3]=st; L.p[4]=pd;
                    L.prevC = prevC;
                    L.wn = (size_t)KH * KW * prevC * Cout;
                    size_t bn = Cout;
                    L.w=(const float*)(blob+o); o+=(size_t)L.wn*4;
                    L.b=(const float*)(blob+o); o+=bn*4;
                    prevC = Cout; break;
                }
                case 2: { /* FC */
                    int Cout = read_u32(blob + o); o += 4;
                    int inDim = read_u32(blob + o); o += 4;
                    L.p[0]=Cout; L.p[1]=inDim;
                    L.wn = (size_t)inDim * Cout;
                    size_t bn = Cout;
                    L.w=(const float*)(blob+o); o+=(size_t)L.wn*4;
                    L.b=(const float*)(blob+o); o+=bn*4;
                    prevC = Cout; break;
                }
                case 3: break;                                   /* ReLU */
                case 4: {                                        /* MaxPool */
                    int pool=read_u32(blob+o); o+=4;
                    int st=read_u32(blob+o); o+=4;
                    int pd=read_u32(blob+o); o+=4;
                    L.p[0]=pool; L.p[1]=st; L.p[2]=pd; break;
                }
                case 5: break;                                   /* Softmax */
                case 6: {                                        /* BN */
                    int ch=read_u32(blob+o); o+=4;
                    L.p[0]=ch;
                    L.wn=(size_t)ch*4;
                    L.w=(const float*)(blob+o); o+=(size_t)L.wn*4;
                    break;
                }
                default: free(m); free(blob); return nullptr;
            }
        }
    }

    /* ---- 第二遍: 融合 (BN 折叠 + ReLU 融合) ---- */
    std::vector<Layer> out;
    std::vector<char> consumed(num, 0);
    for (int i = 0; i < num; ++i) {
        if (consumed[i]) continue;
        RawLayer& L = raw[i];
        if (L.type == 1 || L.type == 2) {            /* Conv / FC */
            Layer F; std::memset(&F, 0, sizeof(F));
            F.type = L.type; std::memcpy(F.p, L.p, sizeof(L.p));
            F.w = L.w; F.b = L.b; F.act = 0;
            int Cout = (L.type == 1) ? L.p[2] : L.p[0];
            int j = i + 1;
            bool sawReLU = false;
            if (j < num && raw[j].type == 3) { sawReLU = true; j++; }
            if (j < num && raw[j].type == 6 && raw[j].p[0] == Cout) {
                fold_bn_into(F, raw[j], Cout, L.wn, m);   /* BN 折叠进权重 */
                consumed[j] = 1;
            }
            if (sawReLU) { F.act = 1; consumed[i + 1] = 1; }  /* ReLU 融合 (无论 BN 是否折叠) */
            out.push_back(F);
        } else if (L.type == 3) {                 /* 独立 ReLU (未被前层吸收) */
            Layer F; std::memset(&F, 0, sizeof(F)); F.type = 3; out.push_back(F);
        } else if (L.type == 6) {                 /* 独立 BN (通道数不匹配等) */
            Layer F; std::memset(&F, 0, sizeof(F)); F.type = 6; F.p[0]=L.p[0]; F.w=L.w; out.push_back(F);
        } else {                                  /* MaxPool / Softmax */
            Layer F; std::memset(&F, 0, sizeof(F)); F.type=L.type; std::memcpy(F.p, L.p, sizeof(L.p));
            F.w=L.w; F.b=L.b; out.push_back(F);
        }
    }

    m->num = (int)out.size();
    m->layers = (Layer*)malloc(sizeof(Layer) * (out.size() ? out.size() : 1));
    if (!m->layers) { acfun_model_destroy(m); return nullptr; }
    std::memcpy(m->layers, out.data(), sizeof(Layer) * out.size());
    return m;
}

void acfun_model_destroy(AcfunModel* m) {
    if (!m) return;
    for (float* p : m->owned) free(p);
    free(m->buf[0]); free(m->buf[1]);
    free(m->layers);
    free(m->blob);
    free(m);
}

int acfun_model_output_size(AcfunModel* m) { return m ? m->outSize : 0; }

void acfun_model_input_dims(AcfunModel* m, int* Cin, int* Hin, int* Win) {
    if (!m) return;
    if (Cin) *Cin = m->inC;
    if (Hin) *Hin = m->inH;
    if (Win) *Win = m->inW;
}

int acfun_model_infer(AcfunModel* m, const float* input, float* output) {
    if (!m) return 0;
    int N = 1, H = m->inH, W = m->inW, C = m->inC;
    size_t cap = (size_t)N * H * W * C;

    /* 双缓冲 ping-pong: 仅在 ensure_buf 增长时扩容, 循环内不 free, 由 destroy 统一释放 */
    float* cur = ensure_buf(m, 0, cap);
    if (!cur) return 0;
    memcpy(cur, input, cap * sizeof(float));
    m->cur = 0;

    for (int i = 0; i < m->num; ++i) {
        Layer* L = &m->layers[i];
        int nH = H, nW = W, nC = C;
        size_t ncap = 0;
        switch (L->type) {
            case 1: {   /* Conv (可选融合 ReLU) */
                int KH=L->p[0], KW=L->p[1], Cout=L->p[2], st=L->p[3], pd=L->p[4];
                nH=(H+2*pd-KH)/st+1; nW=(W+2*pd-KW)/st+1;
                nC=Cout; ncap=(size_t)N*nH*nW*nC;
                int dst = 1 - m->cur;
                float* outb = ensure_buf(m, dst, ncap);
                if (L->act==1) acfun_conv2d_relu(cur,N,H,W,C,L->w,L->b,KH,KW,Cout,st,pd,outb);
                else           acfun_conv2d    (cur,N,H,W,C,L->w,L->b,KH,KW,Cout,st,pd,outb);
                m->cur=dst; cur=outb;
                break;
            }
            case 2: {   /* FC (可选融合 ReLU) */
                int Cout=L->p[0];
                int inDim=H*W*C;
                nH=1; nW=1; nC=Cout; ncap=(size_t)N*nC;
                int dst = 1 - m->cur;
                float* outb = ensure_buf(m, dst, ncap);
                acfun_fc(cur,N,inDim,L->w,L->b,Cout,outb);
                if (L->act==1) acfun_relu(outb, ncap);
                m->cur=dst; cur=outb;
                break;
            }
            case 3: {   /* ReLU (就地) */
                acfun_relu(cur, cap);
                ncap=cap; break;
            }
            case 4: {   /* MaxPool */
                int pool=L->p[0], st=L->p[1], pd=L->p[2];
                nH=(H+2*pd-pool)/st+1; nW=(W+2*pd-pool)/st+1;
                nC=C; ncap=(size_t)N*nH*nW*nC;
                int dst = 1 - m->cur;
                float* outb = ensure_buf(m, dst, ncap);
                acfun_maxpool(cur,N,H,W,C,pool,st,pd,outb);
                m->cur=dst; cur=outb;
                break;
            }
            case 5: {   /* Softmax (就地) */
                acfun_softmax(cur,1,(int)cap,cur);
                ncap=cap; break;
            }
            case 6: {   /* BN (就地) */
                int ch=L->p[0];
                acfun_batchnorm(cur,(int)cap,ch,L->w,L->w+ch,L->w+2*ch,L->w+3*ch,1e-5f);
                ncap=cap; break;
            }
        }
        H=nH; W=nW; C=nC; cap=ncap;
    }
    m->outSize=(int)cap;
    memcpy(output, cur, cap*sizeof(float));
    return (int)cap;
}
