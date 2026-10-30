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
 *     FC   : u32 Cout               | float w[Cin*Cout]          | float b[Cout]
 *     ReLU : (无参数)
 *     MaxP : u32 pool,stride,pad
 *     SoftM: (无参数)
 *     BN   : u32 ch | float mean[ch] var[ch] gamma[ch] beta[ch]
 *   其中 Conv 的 Cin 取上一层输出通道。 */
#include "acfun.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

struct Layer {
    int type;
    int p[8];
    const float* w;
    const float* b;
};

struct AcfunModel {
    int inC, inH, inW;
    int num;
    Layer* layers;
    unsigned char* blob;
    int outSize;
};

static unsigned read_u32(const unsigned char* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
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
    size_t off = 6;
    /* version */ off += 4;
    int inC = read_u32(blob + off); off += 4;
    int inH = read_u32(blob + off); off += 4;
    int inW = read_u32(blob + off); off += 4;
    int num = read_u32(blob + off); off += 4;

    AcfunModel* m = (AcfunModel*)calloc(1, sizeof(AcfunModel));
    if (!m) { free(blob); return nullptr; }
    m->inC = inC; m->inH = inH; m->inW = inW; m->num = num;
    m->blob = blob;
    m->layers = (Layer*)calloc(num ? num : 1, sizeof(Layer));
    if (!m->layers) { free(blob); free(m); return nullptr; }

    int prevC = inC;
    for (int i = 0; i < num; ++i) {
        Layer* L = &m->layers[i];
        L->type = read_u32(blob + off); off += 4;
        switch (L->type) {
            case 1: { /* Conv */
                int KH = read_u32(blob + off); off += 4;
                int KW = read_u32(blob + off); off += 4;
                int Cout = read_u32(blob + off); off += 4;
                int st = read_u32(blob + off); off += 4;
                int pd = read_u32(blob + off); off += 4;
                L->p[0] = KH; L->p[1] = KW; L->p[2] = Cout; L->p[3] = st; L->p[4] = pd;
                size_t wn = (size_t)KH * KW * prevC * Cout;
                size_t bn = Cout;
                L->w = (const float*)(blob + off); off += wn * 4;
                L->b = (const float*)(blob + off); off += bn * 4;
                prevC = Cout;
                break;
            }
            case 2: { /* FC: u32 Cout, u32 inDim, w[inDim*Cout], b[Cout] */
                int Cout = read_u32(blob + off); off += 4;
                int inDim = read_u32(blob + off); off += 4;
                L->p[0] = Cout; L->p[1] = inDim;
                size_t wn = (size_t)inDim * Cout;
                size_t bn = Cout;
                L->w = (const float*)(blob + off); off += wn * 4;
                L->b = (const float*)(blob + off); off += bn * 4;
                prevC = Cout;
                break;
            }
            case 3: break;                                /* ReLU */
            case 4: {                                     /* MaxPool */
                int pool = read_u32(blob + off); off += 4;
                int st = read_u32(blob + off); off += 4;
                int pd = read_u32(blob + off); off += 4;
                L->p[0] = pool; L->p[1] = st; L->p[2] = pd;
                break;
            }
            case 5: break;                                /* Softmax */
            case 6: {                                     /* BN */
                int ch = read_u32(blob + off); off += 4;
                L->p[0] = ch;
                size_t n = (size_t)ch * 4;
                L->w = (const float*)(blob + off); /* mean|var|gamma|beta 连续存放 */
                off += n * 4;
                break;
            }
            default: free(m->layers); free(blob); free(m); return nullptr;
        }
    }
    return m;
}

void acfun_model_destroy(AcfunModel* m) {
    if (!m) return;
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
    float* cur = (float*)malloc(cap * sizeof(float));
    float* nxt = nullptr;
    if (!cur) return 0;
    memcpy(cur, input, cap * sizeof(float));

    for (int i = 0; i < m->num; ++i) {
        Layer* L = &m->layers[i];
        int nH = H, nW = W, nC = C;
        size_t ncap = 0;
        switch (L->type) {
            case 1: {
                int KH = L->p[0], KW = L->p[1], Cout = L->p[2], st = L->p[3], pd = L->p[4];
                nH = (H + 2 * pd - KH) / st + 1;
                nW = (W + 2 * pd - KW) / st + 1;
                nC = Cout; ncap = (size_t)N * nH * nW * nC;
                nxt = (float*)malloc(ncap * sizeof(float));
                if (!nxt) { free(cur); return 0; }
                acfun_conv2d(cur, N, H, W, C, L->w, L->b, KH, KW, Cout, st, pd, nxt);
                break;
            }
            case 2: {
                int Cout = L->p[0];
                int inDim = H * W * C;        /* 展平空间+通道维 */
                nH = 1; nW = 1; nC = Cout; ncap = (size_t)N * nC;
                nxt = (float*)malloc(ncap * sizeof(float));
                if (!nxt) { free(cur); return 0; }
                acfun_fc(cur, N, inDim, L->w, L->b, Cout, nxt);
                break;
            }
            case 3: {
                ncap = cap; nxt = (float*)malloc(ncap * sizeof(float));
                if (!nxt) { free(cur); return 0; }
                memcpy(nxt, cur, ncap * sizeof(float));
                acfun_relu(nxt, ncap);
                break;
            }
            case 4: {
                int pool = L->p[0], st = L->p[1], pd = L->p[2];
                nH = (H + 2 * pd - pool) / st + 1;
                nW = (W + 2 * pd - pool) / st + 1;
                nC = C; ncap = (size_t)N * nH * nW * nC;
                nxt = (float*)malloc(ncap * sizeof(float));
                if (!nxt) { free(cur); return 0; }
                acfun_maxpool(cur, N, H, W, C, pool, st, pd, nxt);
                break;
            }
            case 5: {
                ncap = cap; nxt = (float*)malloc(ncap * sizeof(float));
                if (!nxt) { free(cur); return 0; }
                acfun_softmax(cur, 1, (int)cap, nxt);
                break;
            }
            case 6: {
                int ch = L->p[0];
                ncap = cap; nxt = (float*)malloc(ncap * sizeof(float));
                if (!nxt) { free(cur); return 0; }
                memcpy(nxt, cur, ncap * sizeof(float));
                acfun_batchnorm(nxt, (int)cap, ch, L->w, L->w + ch, L->w + 2 * ch, L->w + 3 * ch, 1e-5f);
                break;
            }
        }
        free(cur);
        cur = nxt; nxt = nullptr;
        H = nH; W = nW; C = nC; cap = ncap;
    }
    m->outSize = (int)cap;
    memcpy(output, cur, cap * sizeof(float));
    free(cur);
    return (int)cap;
}
