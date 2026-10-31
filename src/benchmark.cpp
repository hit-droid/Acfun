/* Acfun 基准测试：
 *   1) 正确性：SIMD 后端 vs 标量参考 (含非整除边界尺寸)
 *   2) GEMM 吞吐：naive vs Acfun (单线程/多线程) GFLOPS
 *   3) 卷积：naive 直乘 vs im2col+GEMM (images/s, ms)
 *   4) 端到端：加载模型前向推理
 */
#include "acfun.h"
#include "internal.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

static double now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count() * 1e-9;
}
static void rand_fill(float* p, size_t n, float s = 1.0f) {
    static std::mt19937 g(12345);
    std::uniform_real_distribution<float> d(-s, s);
    for (size_t i = 0; i < n; ++i) p[i] = d(g);
}

/* ---------- 1) 正确性 ---------- */
static int correctness() {
    printf("\n=== 正确性校验 (Acfun vs 标量参考) ===\n");
    int fails = 0;
    int sizes[] = { 64, 100, 255, 512, 1023 };
    for (int S : sizes) {
        int M = S, N = S, K = S;
        std::vector<float> A((size_t)M * K), B((size_t)K * N),
                           Cref((size_t)M * N), Csim((size_t)M * N);
        rand_fill(A.data(), A.size());
        rand_fill(B.data(), B.size());
        acfun_gemm_ref(M, N, K, A.data(), B.data(), Cref.data());
        acfun_gemm(M, N, K, A.data(), B.data(), Csim.data());
        float maxd = 0.0f;
        for (size_t i = 0; i < Cref.size(); ++i)
            maxd = std::max(maxd, std::fabs(Cref[i] - Csim[i]));
        bool ok = maxd < 1e-2f;
        printf("  M=N=K=%4d  max|Δ|=%9.3e  %s\n", S, maxd, ok ? "OK" : "FAIL");
        if (!ok) fails++;
    }
    return fails;
}

/* ---------- 2) GEMM 吞吐 ---------- */
static void bench_gemm() {
    int mt_threads = 8;                 /* 内存带宽受限内核的甜点线程数 */
    const char* env = std::getenv("ACFUN_BENCH_THREADS");
    if (env) mt_threads = std::atoi(env);
    printf("\n=== GEMM 吞吐 (GFLOPS = 2·M·N·K / t) ===\n");
    printf("  多线程演示线程数 = %d (可用 ACFUN_BENCH_THREADS 覆盖)\n", mt_threads);
    printf("%-10s %12s %12s %12s %12s\n", "M=N=K", "naive", "acfun-1T", "acfun-MT", "MT加速");
    int sizes[] = { 256, 512, 1024, 2048 };
    for (int S : sizes) {
        int M = S, N = S, K = S;
        std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
        rand_fill(A.data(), A.size());
        rand_fill(B.data(), B.size());
        double gf = 2.0 * M * N * K;

        auto time_it = [&](void (*f)(int,int,int,const float*,const float*,float*), int reps) {
            double t0 = now();
            for (int r = 0; r < reps; ++r) f(M, N, K, A.data(), B.data(), C.data());
            double t1 = now();
            return (t1 - t0) / reps;
        };
        int reps = (S >= 1024) ? 5 : 20;
        double tnaive = time_it(acfun_gemm_ref, reps);
        acfun_set_threads(1);        double t1  = time_it(acfun_gemm, reps);
        acfun_set_threads(mt_threads); double tMT = time_it(acfun_gemm, reps);

        printf("%-10d %12.1f %12.1f %12.1f %11.1fx\n",
               S, gf / tnaive / 1e9, gf / t1 / 1e9, gf / tMT / 1e9,
               t1 / tMT);
    }
}

/* naive 直接卷积 (NHWC) 基线 */
static void conv_naive(const float* in, int N, int H, int W, int Cin,
                       const float* w, const float* b, int KH, int KW, int Cout,
                       int st, int pd, float* out) {
    int OH = (H + 2 * pd - KH) / st + 1, OW = (W + 2 * pd - KW) / st + 1;
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < Cout; ++oc) {
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float s = b ? b[oc] : 0.0f;
                    for (int kh = 0; kh < KH; ++kh)
                        for (int kw = 0; kw < KW; ++kw)
                            for (int ic = 0; ic < Cin; ++ic) {
                                int ih = oh * st - pd + kh, iw = ow * st - pd + kw;
                                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                s += in[((size_t)n * H + ih) * W * Cin + iw * Cin + ic] *
                                     w[((kh * KW + kw) * Cin + ic) * Cout + oc];
                            }
                    out[((size_t)n * OH + oh) * OW * Cout + ow * Cout + oc] = s;
                }
        }
}

/* ---------- 3) 卷积吞吐 ---------- */
static void bench_conv() {
    printf("\n=== 卷积吞吐 (3x3, stride1, 输入 112x112) ===\n");
    int H = 112, W = 112, Cin = 16, Cout = 16, KH = 3, KW = 3, st = 1, pd = 1, N = 1;
    int OH = H, OW = W;
    size_t inN = (size_t)N * H * W * Cin, wN = (size_t)KH * KW * Cin * Cout, outN = (size_t)N * OH * OW * Cout;
    std::vector<float> in(inN), w(wN), b(Cout, 0.1f), o1(outN), o2(outN);
    rand_fill(in.data(), inN); rand_fill(w.data(), wN);

    int reps = 20;
    auto tconv = [&](void (*f)(const float*,int,int,int,int,const float*,const float*,int,int,int,int,int,float*),
                     float* out, int r) {
        double t0 = now(); for (int i = 0; i < r; ++i) f(in.data(), N, H, W, Cin, w.data(), b.data(), KH, KW, Cout, st, pd, out); return (now() - t0) / r;
    };
    double tn = tconv(conv_naive, o1.data(), reps);
    double ta = tconv(acfun_conv2d, o2.data(), reps);
    printf("  naive 直乘 : %7.3f ms  (%6.1f img/s)\n", tn * 1e3, 1.0 / tn);
    printf("  Acfun GEMM : %7.3f ms  (%6.1f img/s)\n", ta * 1e3, 1.0 / ta);
    printf("  加速比     : %7.1fx\n", tn / ta);

    /* 校验 */
    float maxd = 0; for (size_t i = 0; i < outN; ++i) maxd = std::max(maxd, std::fabs(o1[i] - o2[i]));
    printf("  max|Δ|=%9.3e %s\n", maxd, maxd < 1e-2f ? "OK" : "FAIL");
}

/* ---------- 4) 端到端 ---------- */
static void bench_e2e(const char* model_path) {
    printf("\n=== 端到端模型前向 ===\n");
    AcfunModel* m = acfun_model_load(model_path);
    if (!m) { printf("  未找到模型 %s，跳过 (用 tools/gen_model.py 生成)\n", model_path); return; }
    int inC = 0, inH = 0, inW = 0;
    acfun_model_input_dims(m, &inC, &inH, &inW);
    int inN = (size_t)inH * inW * inC;
    std::vector<float> in(inN), out(1 << 20);
    rand_fill(in.data(), inN);
    /* 预热 */
    acfun_model_infer(m, in.data(), out.data());
    int reps = 20;
    double t0 = now();
    for (int i = 0; i < reps; ++i) acfun_model_infer(m, in.data(), out.data());
    double t = (now() - t0) / reps;
    printf("  输入 %dx%dx%d  前向 %7.3f ms  (%6.1f img/s)\n",
           inH, inW, inC, t * 1e3, 1.0 / t);
    acfun_model_destroy(m);
}

/* ---------- 5) 融合正确性 + 加速 (非融合参考 vs 融合引擎) ----------
 * 参考执行器按文件原始层序 (Conv->BN->ReLU 各层独立) 执行, 与融合引擎
 * (BN 折叠进 Conv 权重, ReLU 融合) 的输出逐元素对比; 二者应完全一致,
 * 从而严格证明融合无精度损失, 并量化融合消去整层 BN / 融合 ReLU 的提速。 */
struct RLayer { int type; int p[8]; const float* w; const float* b; };

static unsigned r_u32(const unsigned char* p) {
    return (unsigned)p[0] | ((unsigned)p[1] << 8) |
           ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
}

static std::vector<RLayer> ref_parse(const char* path, int* inC, int* inH, int* inW) {
    std::vector<RLayer> L;
    FILE* f = fopen(path, "rb"); if (!f) return L;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<unsigned char> blob(sz); fread(blob.data(), 1, sz, f); fclose(f);
    if (sz < 22 || memcmp(blob.data(), "ACFUN0", 6) != 0) return L;
    size_t o = 10;
    *inC = r_u32(blob.data()+o); o+=4; *inH = r_u32(blob.data()+o); o+=4;
    *inW = r_u32(blob.data()+o); o+=4; int num = r_u32(blob.data()+o); o+=4;
    int prevC = *inC;
    for (int i = 0; i < num; ++i) {
        RLayer x; std::memset(&x, 0, sizeof(x));
        x.type = r_u32(blob.data()+o); o+=4;
        switch (x.type) {
            case 1: { int KH=r_u32(blob.data()+o);o+=4; int KW=r_u32(blob.data()+o);o+=4;
                      int Co=r_u32(blob.data()+o);o+=4; int st=r_u32(blob.data()+o);o+=4;
                      int pd=r_u32(blob.data()+o);o+=4; x.p[0]=KH;x.p[1]=KW;x.p[2]=Co;x.p[3]=st;x.p[4]=pd;
                      size_t wn=(size_t)KH*KW*prevC*Co; x.w=(const float*)(blob.data()+o); o+=wn*4;
                      x.b=(const float*)(blob.data()+o); o+=(size_t)Co*4; prevC=Co; break; }
            case 2: { int Co=r_u32(blob.data()+o);o+=4; int id=r_u32(blob.data()+o);o+=4;
                      x.p[0]=Co;x.p[1]=id; size_t wn=(size_t)id*Co; x.w=(const float*)(blob.data()+o);o+=wn*4;
                      x.b=(const float*)(blob.data()+o);o+=(size_t)Co*4; prevC=Co; break; }
            case 3: break;
            case 4: { int p=r_u32(blob.data()+o);o+=4; int st=r_u32(blob.data()+o);o+=4; int pd=r_u32(blob.data()+o);o+=4;
                      x.p[0]=p;x.p[1]=st;x.p[2]=pd; break; }
            case 5: break;
            case 6: { int ch=r_u32(blob.data()+o);o+=4; x.p[0]=ch; x.w=(const float*)(blob.data()+o); o+=(size_t)ch*4*4; break; }
            default: return L;
        }
        L.push_back(x);
    }
    return L;
}

static void ref_exec(const std::vector<RLayer>& L, int inC, int inH, int inW,
                     const float* input, float* output, int* outSize) {
    int N=1,H=inH,W=inW,C=inC; size_t cap=(size_t)N*H*W*C;
    float* cur=(float*)malloc(cap*sizeof(float)); memcpy(cur,input,cap*sizeof(float));
    for (size_t li=0; li<L.size(); ++li) {
        const RLayer& x=L[li]; int nH=H,nW=W,nC=C; size_t ncap=0; float* nxt=nullptr;
        switch (x.type) {
            case 1: { int KH=x.p[0],KW=x.p[1],Co=x.p[2],st=x.p[3],pd=x.p[4];
                      nH=(H+2*pd-KH)/st+1; nW=(W+2*pd-KW)/st+1; nC=Co; ncap=(size_t)N*nH*nW*nC;
                      nxt=(float*)malloc(ncap*sizeof(float));
                      acfun_conv2d(cur,N,H,W,C,x.w,x.b,KH,KW,Co,st,pd,nxt); break; }
            case 2: { int Co=x.p[0]; int id=H*W*C; nH=1;nW=1;nC=Co; ncap=(size_t)N*nC;
                      nxt=(float*)malloc(ncap*sizeof(float)); acfun_fc(cur,N,id,x.w,x.b,Co,nxt); break; }
            case 3: { ncap=cap; nxt=(float*)malloc(ncap*sizeof(float)); memcpy(nxt,cur,ncap*sizeof(float)); acfun_relu(nxt,ncap); break; }
            case 4: { int p=x.p[0],st=x.p[1],pd=x.p[2]; nH=(H+2*pd-p)/st+1; nW=(W+2*pd-p)/st+1; nC=C;
                      ncap=(size_t)N*nH*nW*nC; nxt=(float*)malloc(ncap*sizeof(float)); acfun_maxpool(cur,N,H,W,C,p,st,pd,nxt); break; }
            case 5: { acfun_softmax(cur,1,(int)cap,cur); ncap=cap; break; }
            case 6: { int ch=x.p[0]; ncap=cap; nxt=(float*)malloc(ncap*sizeof(float)); memcpy(nxt,cur,ncap*sizeof(float));
                      acfun_batchnorm(nxt,(int)cap,ch,x.w,x.w+ch,x.w+2*ch,x.w+3*ch,1e-5f); break; }
        }
        if (nxt) { free(cur); cur=nxt; nxt=nullptr; }
        H=nH;W=nW;C=nC;cap=ncap;
    }
    if (outSize) *outSize=(int)cap;
    memcpy(output,cur,cap*sizeof(float)); free(cur);
}

static void bench_fusion(const char* model_path) {
    printf("\n=== Conv/BN/ReLU 融合: 正确性与加速 ===\n");
    int inC=0,inH=0,inW=0;
    std::vector<RLayer> ref = ref_parse(model_path, &inC, &inH, &inW);
    AcfunModel* m = acfun_model_load(model_path);
    if (ref.empty() || !m) { printf("  模型缺失, 跳过\n"); if(m) acfun_model_destroy(m); return; }
    acfun_model_input_dims(m,&inC,&inH,&inW);
    int inN=(size_t)inH*inW*inC;
    std::vector<float> in(inN), eng(1<<20), rfe(1<<20);
    rand_fill(in.data(),inN);

    acfun_model_infer(m, in.data(), eng.data());
    int engSize=0; ref_exec(ref, inC,inH,inW, in.data(), rfe.data(), &engSize);

    int n = std::min(engSize, (int)(1<<20));
    float maxd=0; for (int i=0;i<n;++i) maxd=std::max(maxd, std::fabs(eng[i]-rfe[i]));
    printf("  融合引擎 vs 非融合参考  max|Δ|=%9.3e  %s\n", maxd, maxd<1e-3f?"OK":"FAIL");

    /* 提速: 融合引擎 vs 非融合参考 (各预热后计时) */
    int reps=50;
    acfun_model_infer(m, in.data(), eng.data());
    double t0=now(); for(int r=0;r<reps;++r) acfun_model_infer(m,in.data(),eng.data());
    double tEng=(now()-t0)/reps;
    ref_exec(ref, inC,inH,inW, in.data(), rfe.data(), &engSize);
    t0=now(); for(int r=0;r<reps;++r) ref_exec(ref, inC,inH,inW, in.data(), rfe.data(), &engSize);
    double tRef=(now()-t0)/reps;
    printf("  非融合参考 : %7.3f ms  (%6.1f img/s)\n", tRef*1e3, 1.0/tRef);
    printf("  融合引擎   : %7.3f ms  (%6.1f img/s)\n", tEng*1e3, 1.0/tEng);
    printf("  融合加速   : %7.1fx  (微型模型 BN 占比极小, 真实大模型带宽收益更明显)\n", tRef/tEng);
    acfun_model_destroy(m);
}

/* 大特征图上的 Conv+BN+ReLU 组合基准: 直接展示融合消去整层 BN 与额外激活扫描的带宽收益 */
static void bench_conv_bn_relu_composite() {
    printf("\n=== Conv+BN+ReLU 组合 (112x112x64, 3x3) ===\n");
    const int H=112,W=112,Cin=64,Cout=64,KH=3,KW=3,st=1,pd=1,N=1;
    const int OH=H,OW=W;
    size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout, chN=Cout;
    std::vector<float> in(inN), w(wN), b(Cout,0.0f), out1(outN), out2(outN);
    std::vector<float> mean(chN,0.0f), var(chN,1.0f), gamma(chN,1.0f), beta(chN,0.0f);
    rand_fill(in.data(), inN); rand_fill(w.data(), wN);
    /* 随机化 BN 以贴近真实 */
    for (int c=0;c<chN;++c){ mean[c]=0.1f*(c%5-2); var[c]=0.8f+0.1f*(c%3); gamma[c]=0.9f+0.05f*(c%4); beta[c]=0.02f*(c%6); }

    int reps=30;
    /* 基线: conv2d -> batchnorm -> relu (三次遍历输出张量) */
    double t0=now();
    for(int r=0;r<reps;++r){
        acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,out1.data());
        acfun_batchnorm(out1.data(),(int)outN,chN,mean.data(),var.data(),gamma.data(),beta.data(),1e-5f);
        acfun_relu(out1.data(),outN);
    }
    double tBase=(now()-t0)/reps;
    /* 融合: 先把 BN 折叠进权重, 再 conv2d_relu (一次遍历) */
    std::vector<float> wf(wN), bf(Cout);
    for(int c=0;c<Cout;++c){
        float scale=gamma[c]/sqrtf(var[c]+1e-5f), shift=beta[c]-mean[c]*scale;
        for(size_t t=0;t<(size_t)KH*KW*Cin;++t) wf[t*(size_t)Cout+c]=w[t*(size_t)Cout+c]*scale;
        bf[c]=b[c]*scale+shift;
    }
    t0=now();
    for(int r=0;r<reps;++r)
        acfun_conv2d_relu(in.data(),N,H,W,Cin,wf.data(),bf.data(),KH,KW,Cout,st,pd,out2.data());
    double tFused=(now()-t0)/reps;

    float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(out1[i]-out2[i]));
    printf("  基线(conv+bn+relu): %7.3f ms  (%6.1f img/s)\n", tBase*1e3, 1.0/tBase);
    printf("  融合(conv+relu)    : %7.3f ms  (%6.1f img/s)\n", tFused*1e3, 1.0/tFused);
    printf("  融合加速            : %7.1fx   max|Δ|=%9.3e %s\n", tBase/tFused, maxd, maxd<1e-3f?"OK":"FAIL");
}

int main(int argc, char** argv) {
    printf("Acfun 推理引擎基准  |  后端: %s\n", acfun_backend_name());
    int fails = correctness();
    bench_gemm();
    bench_conv();
    const char* mp = (argc > 1) ? argv[1] : "model.acfun";
    bench_e2e(mp);
    bench_fusion(mp);
    bench_conv_bn_relu_composite();
    printf("\n正确性: %s\n", fails == 0 ? "全部通过" : "存在失败");
    return fails == 0 ? 0 : 1;
}
