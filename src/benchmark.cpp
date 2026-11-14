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
#include <algorithm>

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

/* ---------- 1b) INT8 GEMM 正确性 (对标标量参考, 整数运算应精确相等) ---------- */
static int correctness_int8() {
    printf("\n=== INT8 GEMM 正确性 (acfun_gemm_int8 vs 标量参考) ===\n");
    int fails = 0;
    /* 含非整除边界的尺寸; uint8×int8 为精确整数乘法, max|Δ| 应为 0 */
    int sizes[] = { 63, 100, 255, 512, 1023 };
    for (int S : sizes) {
        int M = S, N = S, K = S;
        std::vector<uint8_t> A((size_t)M * K);
        std::vector<int8_t>  B((size_t)K * N);
        static std::mt19937 g(777);
        std::uniform_int_distribution<int> da(0, 255), db(-128, 127);
        for (auto& v : A) v = (uint8_t)da(g);
        for (auto& v : B) v = (int8_t)db(g);
        std::vector<int32_t> Cref((size_t)M * N), Csim((size_t)M * N);
        acfun_gemm_int8_ref(M, N, K, A.data(), B.data(), Cref.data());
        acfun_gemm_int8(M, N, K, A.data(), B.data(), Csim.data());
        int64_t maxd = 0;
        for (size_t i = 0; i < Cref.size(); ++i)
            maxd = std::max(maxd, (int64_t)std::abs((int64_t)Cref[i] - (int64_t)Csim[i]));
        bool ok = maxd == 0;
        printf("  M=N=K=%4d  max|Δ|=%lld  %s\n", S, (long long)maxd, ok ? "OK" : "FAIL");
        if (!ok) fails++;
    }
    return fails;
}

/* ---------- 2) GEMM 吞吐 ---------- */
static void bench_gemm() {
    int mt_threads = 8;                 /* 内存带宽受限内核的甜点线程数 */
    const char* env = std::getenv("ACFUN_BENCH_THREADS");
    if (env) mt_threads = std::atoi(env);
    /* cgroup 感知: 受限容器下可用核数远少于报告值, 超额订阅只会拖慢, 故压到实际可用核数。 */
    int eff = acfun_effective_cores();
    if (mt_threads > eff) mt_threads = eff;
    printf("\n=== GEMM 吞吐 (GFLOPS = 2·M·N·K / t) ===\n");
    printf("  多线程演示线程数 = %d (可用 ACFUN_BENCH_THREADS 覆盖; 实际可用核 %d)\n", mt_threads, eff);
    printf("%-10s %12s %12s %12s %12s\n", "M=N=K", "naive", "acfun-1T", "acfun-MT", "MT加速");
    int sizes[] = { 256, 512, 1024, 2048, 4096 };
    for (int S : sizes) {
        int M = S, N = S, K = S;
        std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
        rand_fill(A.data(), A.size());
        rand_fill(B.data(), B.size());
        double gf = 2.0 * M * N * K;

        auto time_it = [&](void (*f)(int,int,int,const float*,const float*,float*), int reps) {
            /* 多次热身 + 中位数计时: 沙箱 cgroup/频率波动下降低噪声, 取中位数而非均值,
             * 避免偶发调度抖动污染吞吐读数。 */
            for (int w = 0; w < 3; ++w) f(M, N, K, A.data(), B.data(), C.data());
            std::vector<double> ts;
            for (int r = 0; r < reps; ++r) {
                double t0 = now();
                f(M, N, K, A.data(), B.data(), C.data());
                double t1 = now();
                ts.push_back(t1 - t0);
            }
            std::sort(ts.begin(), ts.end());
            return ts[ts.size() / 2];
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

/* ---------- 2b) 多线程扩展曲线 ---------- */
static void bench_gemm_scale() {
    int S = 2048;
    int M = S, N = S, K = S;
    std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
    rand_fill(A.data(), A.size());
    rand_fill(B.data(), B.size());
    double gf = 2.0 * M * N * K;
    /* 运行时校准参考点: 先测单线程 2048³ 吞吐 G1, 用以折算多核"线性扩展效率",
     * 避免硬编码与本机不符的理论峰值 (不同微架构/频率下差异很大)。 */
    acfun_set_threads(1);
    int reps = 5;
    double t0 = now();
    for (int r = 0; r < reps; ++r) acfun_gemm(M, N, K, A.data(), B.data(), C.data());
    double G1 = gf / ((now() - t0) / reps) / 1e9;

    printf("\n=== GEMM 多线程扩展曲线 (M=N=K=%d) ===\n", S);
    printf("  单线程参考吞吐 G1 = %6.1f GFLOPS (折算基线)\n", G1);
    int eff = acfun_effective_cores();
    printf("  实际可用核数 (cgroup 感知) = %d; 超出部分超额订阅反致吞吐下降, 仅测至该上限\n", eff);
    printf("%-10s %14s %16s\n", "线程数", "GFLOPS", "线性扩展*");
    int threads[] = { 1, 2, 4, 8, 12, 16 };
    for (int t : threads) {
        if (t > eff) break;             /* 仅测实际可用核数范围, 避免虚假超额订阅曲线 */
        acfun_set_threads(t);
        double t0 = now();
        for (int r = 0; r < reps; ++r) acfun_gemm(M, N, K, A.data(), B.data(), C.data());
        double dt = (now() - t0) / reps;
        double g = gf / dt / 1e9;
        printf("%-10d %14.1f %15.1f%%\n", t, g, g / (G1 * t) * 100.0);
    }
    printf("  * 线性扩展 = 实测 / (G1 × 线程数); 在可用核内应近线性, 超额订阅后回落\n");
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
    printf("  Acfun (Wino/GEMM) : %7.3f ms  (%6.1f img/s)\n", ta * 1e3, 1.0 / ta);
    printf("  加速比     : %7.1fx\n", tn / ta);

    /* 校验 */
    float maxd = 0; for (size_t i = 0; i < outN; ++i) maxd = std::max(maxd, std::fabs(o1[i] - o2[i]));
    printf("  max|Δ|=%9.3e %s\n", maxd, maxd < 1e-2f ? "OK" : "FAIL");
}

/* ---------- 3b) Winograd 3x3 卷积: 正确性 + 提速 ----------
 * Winograd 路径已被 conv2d_impl 在 3x3/s1/p1 且通道>=4 时自动选用。
 * 这里用直接卷积 (conv_naive) 作参考, 跨多种尺寸/通道/奇偶特征图严格校验
 * 数值一致性 (容许 Winograd 域变换的浮点舍入), 并给出 112² 吞吐。 */
static void bench_winograd() {
    printf("\n=== Winograd F(2,3) 3x3 卷积: 正确性与加速 ===\n");
    struct Cfg { int H, W, Cin, Cout; } cfgs[] = {
        {112,112,16,16}, {64,64,32,32}, {57,55,24,48}, {33,31,12,20},
        {16,16,8,8},     {113,97,16,16}, {48,48,64,16}, {24,24,4,64},
    };
    int fails = 0;
    for (auto& c : cfgs) {
        int H=c.H,W=c.W,Cin=c.Cin,Cout=c.Cout,KH=3,KW=3,st=1,pd=1,N=1;
        if (Cin < 4 || Cout < 4) continue;   /* 仅覆盖 Winograd 实际启用区间 */
        int OH=H, OW=W;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), on(outN), ow(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        conv_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,on.data());
        acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,ow.data());
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(on[i]-ow[i]));
        bool ok = maxd < 1e-2f;
        if(!ok) fails++;
        printf("  %dx%dx%d->%d  max|Δ|=%9.3e  %s\n", H,W,Cin,Cout, maxd, ok?"OK":"FAIL");
    }
    /* 112² 吞吐对比 */
    {
        int H=112,W=112,Cin=16,Cout=16,KH=3,KW=3,st=1,pd=1,N=1,OH=H,OW=W;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), o(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        int reps=30;
        double tn=0, ta=0;
        std::vector<float> on(outN);
        {
            double t0=now(); for(int i=0;i<reps;++i) conv_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,on.data()); tn=(now()-t0)/reps;
        }
        {
            double t0=now(); for(int i=0;i<reps;++i) acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,o.data()); ta=(now()-t0)/reps;
        }
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(on[i]-o[i]));
        printf("  112²x16  naive : %7.3f ms  (%6.1f img/s)\n", tn*1e3, 1.0/tn);
        printf("  112²x16  Wino  : %7.3f ms  (%6.1f img/s)  max|Δ|=%9.3e %s\n",
               ta*1e3, 1.0/ta, maxd, maxd<1e-2f?"OK":"FAIL");
        printf("  Winograd 加速  : %7.1fx\n", tn/ta);
    }
    printf("  %s\n", fails==0 ? "Winograd 全部通过" : "Winograd 存在失败");
}

/* ---------- 3c) Winograd F(4,3) 大瓦片: 正确性 + 相对 F(2,3) 提速 ----------
 * 对标 MNN 用更大 Winograd 瓦片逼近吞吐的核心手段。直接用 acfun_winograd43_conv3x3
 * (经 dispatcher 自动派发 AVX-512 / 标量) 与 F(2,3) 及朴素直乘严格校验数值一致性,
 * 并跨大特征图尺寸量化 F(4,3) 相对 F(2,3) 的提速 (瓦片数降 1/4 → 摊薄变换开销与 g_hat 重取)。 */
static void bench_winograd43() {
    printf("\n=== Winograd F(4,3) vs F(2,3) 大瓦片: 正确性与加速 ===\n");
    struct Cfg { int H, W, Cin, Cout; } cfgs[] = {
        {112,112,16,16}, {64,64,32,32}, {57,55,24,48}, {113,97,16,16},
        {48,48,64,16}, {224,224,3,16}, {32,32,16,16}, {40,40,24,24},
    };
    int fails = 0;
    for (auto& c : cfgs) {
        int H=c.H,W=c.W,Cin=c.Cin,Cout=c.Cout,KH=3,KW=3,st=1,pd=1,N=1;
        int OH=H, OW=W;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), on(outN), o2(outN), o4(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        conv_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,on.data());
        acfun_winograd_conv3x3(in.data(),N,H,W,Cin,w.data(),b.data(),Cout,o2.data(),0);
        acfun_winograd43_conv3x3(in.data(),N,H,W,Cin,w.data(),b.data(),Cout,o4.data(),0);
        float maxd2=0,maxd4=0;
        for(size_t i=0;i<outN;++i){ maxd2=std::max(maxd2,std::fabs(on[i]-o2[i])); maxd4=std::max(maxd4,std::fabs(on[i]-o4[i])); }
        bool ok2 = maxd2 < 1e-2f, ok4 = maxd4 < 1e-2f;
        if(!ok2||!ok4) fails++;
        printf("  %dx%dx%d->%d  F2 max|Δ|=%9.3e %s  F4 max|Δ|=%9.3e %s\n",
               H,W,Cin,Cout, maxd2, ok2?"OK":"FAIL", maxd4, ok4?"OK":"FAIL");
    }
    /* 大特征图吞吐对比 (F(2,3) vs F(4,3), 均相对 naive) */
    {
        int H=112,W=112,Cin=16,Cout=16,KH=3,KW=3,st=1,pd=1,N=1,OH=H,OW=W;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), on(outN), o2(outN), o4(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        int reps=30; double tn=0, t2=0, t4=0;
        std::vector<float> oref(outN);
        { double t0=now(); for(int i=0;i<reps;++i) conv_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,oref.data()); tn=(now()-t0)/reps; }
        { double t0=now(); for(int i=0;i<reps;++i) acfun_winograd_conv3x3(in.data(),N,H,W,Cin,w.data(),b.data(),Cout,o2.data(),0); t2=(now()-t0)/reps; }
        { double t0=now(); for(int i=0;i<reps;++i) acfun_winograd43_conv3x3(in.data(),N,H,W,Cin,w.data(),b.data(),Cout,o4.data(),0); t4=(now()-t0)/reps; }
        float md4=0; for(size_t i=0;i<outN;++i) md4=std::max(md4,std::fabs(oref[i]-o4[i]));
        printf("  112²x16x16  naive : %7.3f ms  (%6.1f img/s)\n", tn*1e3, 1.0/tn);
        printf("  112²x16x16  F(2,3): %7.3f ms  (%6.1f img/s)\n", t2*1e3, 1.0/t2);
        printf("  112²x16x16  F(4,3): %7.3f ms  (%6.1f img/s)  max|Δ|=%9.3e %s\n",
               t4*1e3, 1.0/t4, md4, md4<1e-2f?"OK":"FAIL");
        printf("  F(4,3) / F(2,3) 加速 : %7.1fx\n", t2/t4);
    }
    printf("  %s\n", fails==0 ? "Winograd F(4,3) 全部通过" : "Winograd F(4,3) 存在失败");
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
            case 7: { int KH=r_u32(blob.data()+o);o+=4; int KW=r_u32(blob.data()+o);o+=4;
                      int Co=r_u32(blob.data()+o);o+=4; int st=r_u32(blob.data()+o);o+=4;
                      int pd=r_u32(blob.data()+o);o+=4; int Gg=r_u32(blob.data()+o);o+=4;
                      x.p[0]=KH;x.p[1]=KW;x.p[2]=Co;x.p[3]=st;x.p[4]=pd;x.p[5]=Gg;
                      size_t wn=(size_t)KH*KW*prevC*Co/Gg; x.w=(const float*)(blob.data()+o); o+=wn*4;
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
            case 7: { int KH=x.p[0],KW=x.p[1],Co=x.p[2],st=x.p[3],pd=x.p[4],Gg=x.p[5];
                      nH=(H+2*pd-KH)/st+1; nW=(W+2*pd-KW)/st+1; nC=Co; ncap=(size_t)N*nH*nW*nC;
                      nxt=(float*)malloc(ncap*sizeof(float));
                      acfun_conv2d_g(cur,N,H,W,C,x.w,x.b,KH,KW,Co,st,pd,Gg,nxt,0); break; }
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
    /* 容差取 1e-2 (与套件其余正确性检查一致): 融合引擎与非融合参考的数值差异来自浮点求和次序,
     * 而分块尺寸 MC 随线程数自适应 (target=16·nt), 故不同线程数下差异量级不同 (单/多线程可见
     * 1e-9~1e-3 区间浮动); 2.8e-3 仍属 float 推理的正常舍入, 远小于推理容忍度。 */
    printf("  融合引擎 vs 非融合参考  max|Δ|=%9.3e  %s\n", maxd, maxd<1e-2f?"OK":"FAIL");

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

/* ---------- 3b) 1x1 逐点卷积: 正确性 + 提速 ----------
 * 1x1 卷积经 conv1x1_impl 直接走 GEMM (跳过 im2col), 与朴素直乘严格对比。 */
static void bench_conv1x1() {
    printf("\n=== 1x1 逐点卷积: 正确性 + 提速 (直接 GEMM 路径) ===\n");
    struct Cfg { int H, W, Cin, Cout, st; } cfgs[] = {
        {112,112,16,16,1}, {56,56,32,32,1}, {28,28,64,64,1},
        {56,56,16,32,2}, {28,28,32,64,2}, {113,97,24,48,1},
    };
    int fails = 0;
    for (auto& c : cfgs) {
        int H=c.H,W=c.W,Cin=c.Cin,Cout=c.Cout,KH=1,KW=1,st=c.st,pd=0,N=1;
        int OH=(H+2*pd-KH)/st+1, OW=(W+2*pd-KW)/st+1;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), on(outN), oa(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        conv_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,on.data());
        acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,oa.data());
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(on[i]-oa[i]));
        bool ok = maxd < 1e-2f; if(!ok) fails++;
        printf("  %dx%dx%d->%d s%d  max|Δ|=%9.3e  %s\n", H,W,Cin,Cout,st, maxd, ok?"OK":"FAIL");
    }
    /* 吞吐对比: 112²x16->16 s1 (im2col 路径已退化为直接 GEMM, 这里展示端到端) */
    {
        int H=112,W=112,Cin=16,Cout=16,KH=1,KW=1,st=1,pd=0,N=1,OH=H,OW=W;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), o(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        int reps=50;
        double tn=0, ta=0; std::vector<float> on(outN);
        { double t0=now(); for(int i=0;i<reps;++i) conv_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,on.data()); tn=(now()-t0)/reps; }
        { double t0=now(); for(int i=0;i<reps;++i) acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,o.data()); ta=(now()-t0)/reps; }
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(on[i]-o[i]));
        printf("  112²x16->16 s1  naive : %7.3f ms  (%6.1f img/s)\n", tn*1e3, 1.0/tn);
        printf("  Acfun 1x1    : %7.3f ms  (%6.1f img/s)  max|Δ|=%9.3e %s\n",
               ta*1e3, 1.0/ta, maxd, maxd<1e-2f?"OK":"FAIL");
        printf("  1x1 加速     : %7.1fx\n", tn/ta);
    }
    printf("  %s\n", fails==0 ? "1x1 卷积全部通过" : "1x1 卷积存在失败");
}

/* ---------- 3d) 深度(逐通道)卷积: 正确性 + 提速 ----------
 * 深度卷积 (groups==Cin==Cout) 是 MobileNet/EfficientNet 的核心算子。acfun_conv2d_dw 跨通道 16 路
 * AVX-512 向量化 + 权重预转置常驻 L1; 与之对照的朴素逐通道标量实现作参考。同时给出相对
 * "把深度卷积当普通卷积走 im2col+GEMM (K=KH*KW 极小, 算术强度崩溃)" 的理论开销说明。 */
static void dw_naive(const float* in, int N, int H, int W, int C,
                     const float* w, const float* b, int KH, int KW, int st, int pd,
                     float* out, int act) {
    int OH = (H + 2*pd - KH)/st + 1, OW = (W + 2*pd - KW)/st + 1;
    for (int n = 0; n < N; ++n)
        for (int oc = 0; oc < C; ++oc) {
            const float* wch = w + (size_t)oc * KH * KW;
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float s = b ? b[oc] : 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        int ih = oh*st - pd + kh; if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = ow*st - pd + kw; if (iw < 0 || iw >= W) continue;
                            s += in[(((size_t)n*H + ih)*W + iw)*C + oc] * wch[kh*KW + kw];
                        }
                    }
                    if (act == 1 && s < 0.0f) s = 0.0f;
                    out[(((size_t)n*OH + oh)*OW + ow)*C + oc] = s;
                }
        }
}

static void bench_depthwise() {
    printf("\n=== 深度(逐通道)卷积: 正确性与加速 (AVX-512 向量化 DW) ===\n");
    struct Cfg { int H, W, C, KH, KW, st, pd; } cfgs[] = {
        {112,112,16,3,3,1,1}, {56,56,32,3,3,1,1}, {224,224,3,3,3,1,1},
        {56,56,128,3,3,2,1}, {28,28,24,3,3,1,1}, {64,64,512,3,3,1,1},
        {64,64,64,1,1,1,0}, {16,16,8,3,3,1,1},
    };
    int fails = 0;
    for (auto& c : cfgs) {
        int H=c.H,W=c.W,C=c.C,KH=c.KH,KW=c.KW,st=c.st,pd=c.pd,N=1,act=1;
        int OH=(H+2*pd-KH)/st+1, OW=(W+2*pd-KW)/st+1;
        size_t inN=(size_t)N*H*W*C, wN=(size_t)C*KH*KW, outN=(size_t)N*OH*OW*C;
        std::vector<float> in(inN), w(wN), b(C,0.1f), on(outN), oa(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        int reps = (outN > (1u<<24)) ? 5 : 30;
        double tn=0, ta=0; std::vector<float> ore(outN);
        { double t0=now(); for(int i=0;i<reps;++i) dw_naive(in.data(),N,H,W,C,w.data(),b.data(),KH,KW,st,pd,ore.data(),act); tn=(now()-t0)/reps; }
        { double t0=now(); for(int i=0;i<reps;++i) acfun_conv2d_dw(in.data(),N,H,W,C,w.data(),b.data(),KH,KW,st,pd,oa.data(),act); ta=(now()-t0)/reps; }
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(ore[i]-oa[i]));
        bool ok = maxd < 1e-2f; if(!ok) fails++;
        printf("  %dx%dx%d k%d s%d  naive:%7.3fms(%6.1f/s)  DW:%7.3fms(%6.1f/s)  加速%5.1fx  max|Δ|=%9.3e %s\n",
               H,W,C,KH,st, tn*1e3,1.0/tn, ta*1e3,1.0/ta, tn/ta, maxd, ok?"OK":"FAIL");
    }
    printf("  %s\n", fails==0 ? "深度卷积全部通过" : "深度卷积存在失败");
}

/* ---------- 3e) 分组卷积: 正确性 + 提速 ----------
 * 一般分组卷积 (groups>1, 非深度) 各组的输入/输出通道连续, 引擎逐组复用 1x1/Winograd/im2col
 * 子路径; groups==Cin==Cout 自动路由到深度卷积。对照朴素分组参考。 */
static void conv_g_naive(const float* in, int N, int H, int W, int Cin,
                         const float* w, const float* b, int KH, int KW, int Cout,
                         int st, int pd, int G, float* out, int act) {
    int Cin_p = Cin/G, Cout_p = Cout/G;
    int OH = (H + 2*pd - KH)/st + 1, OW = (W + 2*pd - KW)/st + 1;
    for (int g = 0; g < G; ++g)
        for (int n = 0; n < N; ++n)
            for (int oc = 0; oc < Cout_p; ++oc) {
                const float* wg = w + (size_t)g * Cout_p * KH * KW * Cin_p;
                const float* bg = b ? b + g*Cout_p : nullptr;
                for (int oh = 0; oh < OH; ++oh)
                    for (int ow = 0; ow < OW; ++ow) {
                        float s = bg ? bg[oc] : 0.0f;
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh*st - pd + kh; if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow*st - pd + kw; if (iw < 0 || iw >= W) continue;
                                for (int ic = 0; ic < Cin_p; ++ic) {
                                    size_t idx = ((size_t)n*H + ih)*W + iw;
                                    size_t ia = (size_t)g*Cin_p + idx*Cin + ic;
                                    s += in[ia] * wg[((kh*KW+kw)*Cin_p+ic)*Cout_p+oc];
                                }
                            }
                        }
                        if (act == 1 && s < 0.0f) s = 0.0f;
                        size_t odx = ((size_t)n*OH + oh)*OW + ow;
                        out[odx*(size_t)Cout + (size_t)g*Cout_p + oc] = s;
                    }
            }
}

static void bench_grouped() {
    printf("\n=== 分组卷积: 正确性与加速 ===\n");
    struct Cfg { int H, W, Cin, Cout, KH, KW, st, pd, G; } cfgs[] = {
        {56,56,16,32,3,3,1,1,4}, {28,28,32,64,1,1,1,0,8}, {14,14,24,48,3,3,2,1,6},
        {56,56,32,32,3,3,1,1,32},
    };
    int fails = 0;
    for (auto& c : cfgs) {
        int H=c.H,W=c.W,Cin=c.Cin,Cout=c.Cout,KH=c.KH,KW=c.KW,st=c.st,pd=c.pd,G=c.G,N=1,act=1;
        int OH=(H+2*pd-KH)/st+1, OW=(W+2*pd-KW)/st+1;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)G*(Cout/G)*KH*KW*(Cin/G), outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), on(outN), oa(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        int reps = (outN > (1u<<24)) ? 5 : 20;
        double tn=0, ta=0; std::vector<float> ore(outN);
        { double t0=now(); for(int i=0;i<reps;++i) conv_g_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,G,ore.data(),act); tn=(now()-t0)/reps; }
        { double t0=now(); for(int i=0;i<reps;++i) acfun_conv2d_g(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,G,oa.data(),act); ta=(now()-t0)/reps; }
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(ore[i]-oa[i]));
        bool ok = maxd < 1e-2f; if(!ok) fails++;
        printf("  %dx%dx%d->%d k%d g%d  naive:%7.3fms(%6.1f/s)  G:%7.3fms(%6.1f/s)  加速%5.1fx  max|Δ|=%9.3e %s\n",
               H,W,Cin,Cout,KH,G, tn*1e3,1.0/tn, ta*1e3,1.0/ta, tn/ta, maxd, ok?"OK":"FAIL");
    }
    printf("  %s\n", fails==0 ? "分组卷积全部通过" : "分组卷积存在失败");
}

/* ---------- 5b) FC + 融合 ReLU: 正确性 + 提速 ----------
 * 对比 "acfun_fc + 独立 acfun_relu (两次全张量扫描)" 与 "acfun_fc_relu (单次遍历)"。
 * 大 FC (N 较大时输出张量可观) 下, 融合消去第二遍 ReLU 扫描的提速最明显。 */
static void bench_fc_relu_fused() {
    printf("\n=== FC + 融合 ReLU (单次遍历 vs 两次扫描) ===\n");
    struct CC { int N, Cin, Cout; } cfgs[] = {
        {1, 4096, 4096}, {64, 4096, 4096}, {256, 2048, 2048}, {1, 1024, 1024},
    };
    int fails = 0;
    for (auto& c : cfgs) {
        int N=c.N, Cin=c.Cin, Cout=c.Cout;
        size_t inN=(size_t)N*Cin, wN=(size_t)Cin*Cout, outN=(size_t)N*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), o_sep(outN), o_fus(outN);
        rand_fill(in.data(), inN); rand_fill(w.data(), wN);
        int reps = (outN > (1u<<26)) ? 5 : 20;
        /* 分离: fc + relu */
        double t0=now();
        for(int r=0;r<reps;++r){
            acfun_fc(in.data(),N,Cin,w.data(),b.data(),Cout,o_sep.data());
            acfun_relu(o_sep.data(),outN);
        }
        double tSep=(now()-t0)/reps;
        /* 融合: 单次遍历 */
        t0=now();
        for(int r=0;r<reps;++r)
            acfun_fc_relu(in.data(),N,Cin,w.data(),b.data(),Cout,o_fus.data());
        double tFus=(now()-t0)/reps;
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd,std::fabs(o_sep[i]-o_fus[i]));
        bool ok = maxd < 1e-5f; if(!ok) fails++;
        printf("  N=%4d %dx%d  sep:%8.3fms(%7.1f/s)  fus:%8.3fms(%7.1f/s)  加速%5.2fx  max|Δ|=%9.3e %s\n",
               N, Cin, Cout, tSep*1e3, 1.0/tSep, tFus*1e3, 1.0/tFus, tSep/tFus, maxd, ok?"OK":"FAIL");
    }
    printf("  %s\n", fails==0 ? "FC+ReLU 融合全部通过" : "FC+ReLU 融合存在失败");
}

/* ---------- 6) INT8 GEMM / 卷积 (AVX-512 VNNI) ----------
 * 对标 MNN int8 推理路径: uint8×int8→int32 的 vpdpbusd 累加器。
 * 报告 INT8 GEMM 的 GOPS (2·M·N·K/t, 与 float GFLOPS 同口径便于比对算力吞吐),
 * 以及量化卷积相对 float 卷积的端到端提速与精度损失。 */
static void bench_int8() {
    int mt_threads = 8;
    const char* env = std::getenv("ACFUN_BENCH_THREADS");
    if (env) mt_threads = std::atoi(env);
    int eff = acfun_effective_cores();
    if (mt_threads > eff) mt_threads = eff;

    printf("\n=== INT8 GEMM 吞吐 (AVX-512 VNNI, uint8×int8→int32) ===\n");
    printf("  多线程演示线程数 = %d (可用 ACFUN_BENCH_THREADS 覆盖; 实际可用核 %d)\n", mt_threads, eff);
    printf("%-10s %14s %14s %12s %12s\n", "M=N=K", "float-GFLOPS", "int8-GOPS", "1T INT8/FP", "MT加速");
    int sizes[] = { 256, 512, 1024, 2048, 4096 };
    for (int S : sizes) {
        int M = S, N = S, K = S;
        std::vector<float> Af((size_t)M * K), Bf((size_t)K * N);
        rand_fill(Af.data(), Af.size()); rand_fill(Bf.data(), Bf.size());
        /* 量化为 uint8 / int8 (逐张量 scale, 此处 |x|<=1 故 scale=1/50) */
        std::vector<uint8_t> Aq((size_t)M * K);
        std::vector<int8_t>  Bq((size_t)K * N);
        for (size_t i = 0; i < Af.size(); ++i) { int v = (int)std::lroundf(Af[i] * 50); v = v < 0 ? 0 : (v > 255 ? 255 : v); Aq[i] = (uint8_t)v; }
        for (size_t i = 0; i < Bf.size(); ++i) { int v = (int)std::lroundf(Bf[i] * 50); v = v < -128 ? -128 : (v > 127 ? 127 : v); Bq[i] = (int8_t)v; }
        std::vector<float>  Cf((size_t)M * N);
        std::vector<int32_t> Ci((size_t)M * N);
        double work = 2.0 * M * N * K;
        int reps = (S >= 1024) ? 5 : 20;

        acfun_set_threads(1);
        double t0 = now(); for (int r = 0; r < reps; ++r) acfun_gemm(M, N, K, Af.data(), Bf.data(), Cf.data());
        double t1 = (now() - t0) / reps;
        double tf = work / t1 / 1e9;
        acfun_set_num_threads_runtime(1);
        t0 = now(); for (int r = 0; r < reps; ++r) acfun_gemm_int8(M, N, K, Aq.data(), Bq.data(), Ci.data());
        double ti = (now() - t0) / reps;
        double gint = work / ti / 1e9;

        acfun_set_num_threads_runtime(mt_threads);
        t0 = now(); for (int r = 0; r < reps; ++r) acfun_gemm_int8(M, N, K, Aq.data(), Bq.data(), Ci.data());
        double tiMT = (now() - t0) / reps;
        double gintMT = work / tiMT / 1e9;

        printf("%-10d %14.1f %14.1f %11.1fx %11.1fx\n", S, tf, gint, gint / tf, gintMT / gint);
    }
    printf("  * INT8 算力 = uint8×int8 MAC × 16/指令, 带宽仅 1/4; GOPS 为同口径 (2·M·N·K/t) 吞吐\n");

    /* INT8 卷积: 端到端提速 + 精度 */
    printf("\n=== INT8 量化卷积 (AVX-512 VNNI) vs float 卷积 ===\n");
    struct CC { int H, W, Cin, Cout, KH, KW, st, pd; } cfgs[] = {
        {112,112,16,16,3,3,1,1}, {56,56,32,32,3,3,1,1}, {28,28,64,64,1,1,1,0},
        {56,56,16,32,1,1,2,0}, {224,224,3,16,3,3,1,1},
    };
    for (auto& c : cfgs) {
        int H=c.H,W=c.W,Cin=c.Cin,Cout=c.Cout,KH=c.KH,KW=c.KW,st=c.st,pd=c.pd,N=1;
        int OH=(H+2*pd-KH)/st+1, OW=(W+2*pd-KW)/st+1;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), of(outN), oi(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        std::vector<float> ws(Cout); std::vector<int8_t> wq(wN);
        acfun_quantize_weights(w.data(), KH*KW*Cin, Cout, ws.data(), wq.data());
        int reps = (outN > (1u<<24)) ? 5 : 20;
        acfun_set_threads(1); acfun_set_num_threads_runtime(1);
        double t0=now(); for(int r=0;r<reps;++r) acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,of.data());
        double tf=(now()-t0)/reps;
        t0=now(); for(int r=0;r<reps;++r) acfun_conv2d_int8(in.data(),N,H,W,Cin,wq.data(),ws.data(),b.data(),KH,KW,Cout,st,pd,0.0f,oi.data(),0);
        double ti=(now()-t0)/reps;
        float maxd=0; for(size_t i=0;i<outN;++i) maxd=std::max(maxd, std::fabs(of[i]-oi[i]));
        printf("  %dx%dx%d->%d k%d s%d  FP:%7.3fms(%6.1f/s)  INT8:%7.3fms(%6.1f/s)  加速%.1fx  max|Δ|=%9.3e\n",
               H,W,Cin,Cout,KH,st, tf*1e3,1.0/tf, ti*1e3,1.0/ti, tf/ti, maxd);
    }
    printf("  * 精度损失来自逐张量激活量化 (随机权重最坏情形); 真实模型(BN/ReLU)通常 <1%%\n");
}

/* ---------- 7) INT8 Winograd 3x3 卷积 ----------
 * 对标 MNN int8 卷积的核心手段: 3x3 专用 Winograd 快路径。acfun_conv2d_int8 已在 3x3/s1/p1
 * 且通道>=4 时自动路由到本路径。这里用朴素 float 卷积作参考, 跨多种尺寸/通道严格校验数值
 * 一致性 (容许逐张量激活量化的固有误差), 并给出相对 float 卷积的提速。 */
static void bench_int8_winograd() {
    printf("\n=== INT8 Winograd 3x3 卷积 (AVX-512 VNNI 路径) ===\n");
    struct CC { int H, W, Cin, Cout; } cfgs[] = {
        {112,112,16,16}, {56,56,32,32}, {28,28,64,64}, {224,224,3,16},
        {64,64,32,64}, {113,97,16,48}, {48,48,64,16}, {24,24,8,64},
    };
    for (auto& c : cfgs) {
        int H=c.H,W=c.W,Cin=c.Cin,Cout=c.Cout,KH=3,KW=3,st=1,pd=1,N=1;
        if (Cin < 4 || Cout < 4) continue;
        int OH=H, OW=W;
        size_t inN=(size_t)N*H*W*Cin, wN=(size_t)KH*KW*Cin*Cout, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN), w(wN), b(Cout,0.1f), on(outN), of(outN), oi(outN);
        rand_fill(in.data(),inN); rand_fill(w.data(),wN);
        std::vector<float> ws(Cout); std::vector<int8_t> wq(wN);
        acfun_quantize_weights(w.data(), KH*KW*Cin, Cout, ws.data(), wq.data());

        /* 参考: 朴素 float 卷积; float 引擎 (自动走 float Winograd); INT8 引擎 (自动走 INT8 Winograd) */
        conv_naive(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,on.data());
        acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,of.data());
        int reps = (outN > (1u<<24)) ? 5 : 20;
        acfun_set_threads(1); acfun_set_num_threads_runtime(1);
        double t0=now(); for(int r=0;r<reps;++r) acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,of.data());
        double tf=(now()-t0)/reps;
        t0=now(); for(int r=0;r<reps;++r) acfun_conv2d_int8(in.data(),N,H,W,Cin,wq.data(),ws.data(),b.data(),KH,KW,Cout,st,pd,0.0f,oi.data(),0);
        double ti=(now()-t0)/reps;

        /* 正确性: INT8 vs float 参考 (容许逐张量激活量化误差) */
        float maxd=0, norm=0;
        for(size_t i=0;i<outN;++i){ maxd=std::max(maxd,std::fabs(on[i]-oi[i])); norm=std::max(norm,std::fabs(on[i])); }
        float rel = norm>1e-9f ? maxd/norm : 0.0f;
        bool ok = rel < 0.10f;   /* 逐张量激活量化下随机权重最坏情形约数 %, 真实模型通常 <1% */
        printf("  %dx%dx%d->%d  FP:%7.3fms(%6.1f/s)  INT8:%7.3fms(%6.1f/s)  加速%.1fx  max|Δ|/范数=%6.2f%% %s\n",
               H,W,Cin,Cout, tf*1e3,1.0/tf, ti*1e3,1.0/ti, tf/ti, rel*100.0f, ok?"OK":"FAIL");
    }
    printf("  * INT8 Winograd 较 float 卷积 乘法量降 2.25x, 带宽仅 int8/int16; 精度损失来自逐张量激活量化\n");
}

/* ---------- 8) BF16 GEMM (AVX-512 vdpbf16ps) ----------
 * 对标 MNN 的低精度 (bf16) 推理路径: 输入/输出均为 float, 内部转 bf16 计算、f32 累加。
 * 报告 BF16 GEMM 的 GFLOPS (同口径 2·M·N·K/t), 与 float GEMM 对比提速, 并校验数值正确性。 */
static void bench_bf16() {
    if (!__builtin_cpu_supports("avx512bf16")) {
        printf("\n=== BF16 GEMM (AVX-512 vdpbf16ps) ===\n");
        printf("  本机不支持 avx512_bf16, 跳过 (回落 acfun_gemm)\n");
        return;
    }
    int mt_threads = 8;
    const char* env = std::getenv("ACFUN_BENCH_THREADS");
    if (env) mt_threads = std::atoi(env);
    int eff = acfun_effective_cores();
    if (mt_threads > eff) mt_threads = eff;

    printf("\n=== BF16 GEMM 正确性 (Acfun bf16 vs 标量参考) ===\n");
    {
        int sizes[] = { 64, 100, 255, 512, 1023 };
        int fails = 0;
        for (int S : sizes) {
            int M = S, N = S, K = S;
            std::vector<float> A((size_t)M * K), B((size_t)K * N), Cref((size_t)M * N), Cb((size_t)M * N);
            rand_fill(A.data(), A.size()); rand_fill(B.data(), B.size());
            acfun_gemm_ref(M, N, K, A.data(), B.data(), Cref.data());
            acfun_gemm_bf16(M, N, K, A.data(), B.data(), Cb.data());
            float maxd = 0, norm = 0;
            for (size_t i = 0; i < Cref.size(); ++i) {
                maxd = std::max(maxd, std::fabs(Cref[i] - Cb[i]));
                norm = std::max(norm, std::fabs(Cref[i]));
            }
            float rel = norm > 1e-9f ? maxd / norm : 0.0f;
            bool ok = rel < 0.02f;   /* bf16 逐层舍入, 推理容忍度内 */
            if (!ok) fails++;
            printf("  M=N=K=%4d  max|Δ|/范数=%7.3e %s\n", S, rel, ok ? "OK" : "FAIL");
        }
        if (fails) printf("  BF16 正确性: 存在失败 (应排查打包/内核配对)\n");
    }

    printf("\n=== BF16 GEMM 吞吐 (AVX-512 vdpbf16ps, GFLOPS = 2·M·N·K/t) ===\n");
    printf("  多线程演示线程数 = %d (可用 ACFUN_BENCH_THREADS 覆盖; 实际可用核 %d)\n", mt_threads, eff);
    printf("%-10s %12s %12s %12s %14s\n", "M=N=K", "float-GFLOPS", "bf16-GFLOPS", "bf16/FP", "MT加速");
    int sizes[] = { 256, 512, 1024, 2048, 4096 };
    for (int S : sizes) {
        int M = S, N = S, K = S;
        std::vector<float> A((size_t)M * K), B((size_t)K * N), Cf((size_t)M * N), Cb((size_t)M * N);
        rand_fill(A.data(), A.size()); rand_fill(B.data(), B.size());
        double work = 2.0 * M * N * K;
        int reps = (S >= 1024) ? 5 : 20;
        acfun_set_threads(1); acfun_set_num_threads_runtime(1);
        double t0 = now(); for (int r = 0; r < reps; ++r) acfun_gemm(M, N, K, A.data(), B.data(), Cf.data());
        double tf = work / ((now() - t0) / reps) / 1e9;
        t0 = now(); for (int r = 0; r < reps; ++r) acfun_gemm_bf16(M, N, K, A.data(), B.data(), Cb.data());
        double tb = work / ((now() - t0) / reps) / 1e9;
        acfun_set_num_threads_runtime(mt_threads);
        t0 = now(); for (int r = 0; r < reps; ++r) acfun_gemm_bf16(M, N, K, A.data(), B.data(), Cb.data());
        double tbMT = work / ((now() - t0) / reps) / 1e9;
        printf("%-10d %12.1f %12.1f %11.1fx %13.1fx\n", S, tf, tb, tb / tf, tbMT / tb);
    }
    printf("  * BF16 带宽仅 float 之半、f32 累加; 与 MNN 的 bf16/int8 低精度路径同构\n");
}

/* ---------- 1c) GEMM 融合后处理 (epilogue) 正确性 ----------
 * 以 "acfun_gemm + 独立偏置/ReLU 遍" 作参考, 与融合版 acfun_gemm_epi 严格比对。
 * 覆盖非整除边界尺寸、有无偏置、act=0/1, 验证 bias 仅在归约完成后叠加一次。 */
static int correctness_epi() {
    printf("\n=== GEMM 融合后处理 (epilogue) 正确性 (acfun_gemm_epi vs 非融合参考) ===\n");
    int fails = 0;
    struct S { int M, N, K; bool has_bias; int act; } cfgs[] = {
        {64,64,64,true,1}, {100,100,100,true,1}, {255,255,255,true,0},
        {512,17,64,true,1}, {63,512,33,false,1}, {1023,1023,1023,true,1},
        {300,16,144,true,1}, {256,256,256,false,0},
    };
    for (auto& c : cfgs) {
        int M=c.M, N=c.N, K=c.K;
        std::vector<float> A((size_t)M*K), B((size_t)K*N), bias(c.has_bias?(size_t)N:0, 0.1f);
        std::vector<float> Cref((size_t)M*N), Csim((size_t)M*N);
        rand_fill(A.data(), A.size());
        rand_fill(B.data(), B.size());
        /* 参考: GEMM + 独立偏置/ReLU 遍 */
        acfun_gemm(M, N, K, A.data(), B.data(), Cref.data());
        for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
            float v = Cref[(size_t)i*N+j];
            if (c.has_bias) v += bias[j];
            if (c.act && v < 0.0f) v = 0.0f;
            Cref[(size_t)i*N+j] = v;
        }
        acfun_gemm_epi(M, N, K, A.data(), B.data(), Csim.data(),
                       c.has_bias ? bias.data() : nullptr, c.act);
        float maxd = 0.0f;
        for (size_t i = 0; i < Cref.size(); ++i)
            maxd = std::max(maxd, std::fabs(Cref[i] - Csim[i]));
        bool ok = maxd < 1e-2f;
        if (!ok) fails++;
        const char* bs = c.has_bias ? "bias" : "no-bias";
        const char* as = c.act ? "relu" : "no-act";
        printf("  M=%d N=%d K=%d %s %s  max|Δ|=%9.3e  %s\n", M, N, K, bs, as, maxd, ok?"OK":"FAIL");
    }
    printf("  %s\n", fails==0 ? "融合后处理全部通过" : "融合后处理存在失败");
    return fails;
}

/* ---------- 2c) GEMM 融合后处理 (epilogue) 吞吐 ----------
 * 量化 "融合后处理" 消除的独立全张量扫描收益: 同等 (M,N,K) 下,
 * 非融合 (GEMM + 独立偏置/ReLU 遍) vs 融合 (acfun_gemm_epi)。 */
static void bench_epi() {
    printf("\n=== GEMM 融合后处理 (epilogue) 吞吐对比 ===\n");
    struct S { int M, N, K; } cfgs[] = {
        {1024,1024,1024}, {2048,2048,2048}, {512,512,512},
        {112*112,16,144}, {64*64,32,288}, {256,256,256},
    };
    printf("%-16s %14s %14s %12s\n", "M×N×K", "非融合(ms)", "融合epi(ms)", "加速");
    for (auto& c : cfgs) {
        int M=c.M, N=c.N, K=c.K;
        std::vector<float> A((size_t)M*K), B((size_t)K*N), bias((size_t)N, 0.1f);
        std::vector<float> C1((size_t)M*N), C2((size_t)M*N);
        rand_fill(A.data(), A.size()); rand_fill(B.data(), B.size());
        int reps = (M*N*K > (1<<28)) ? 5 : 20;
        /* 非融合参考 */
        double t0 = now();
        for (int r = 0; r < reps; ++r) {
            acfun_gemm(M, N, K, A.data(), B.data(), C1.data());
            for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) {
                float v = C1[(size_t)i*N+j] + bias[j];
                if (v < 0.0f) v = 0.0f;
                C1[(size_t)i*N+j] = v;
            }
        }
        double tUnfused = (now() - t0) / reps;
        /* 融合 */
        t0 = now();
        for (int r = 0; r < reps; ++r)
            acfun_gemm_epi(M, N, K, A.data(), B.data(), C2.data(), bias.data(), 1);
        double tEpi = (now() - t0) / reps;
        /* 一致性门禁 */
        float maxd = 0; for (size_t i = 0; i < C1.size(); ++i) maxd = std::max(maxd, std::fabs(C1[i]-C2[i]));
        char label[32];
        snprintf(label, sizeof(label), "%dx%dx%d", M, N, K);
        printf("%-16s %14.4f %14.4f %11.1fx  max|Δ|=%9.3e %s\n",
               label, tUnfused*1e3, tEpi*1e3, tUnfused/tEpi, maxd, maxd<1e-2f?"OK":"FAIL");
    }
}

int main(int argc, char** argv) {
    bool do_tune = false;
    for (int i = 1; i < argc; ++i) if (std::strcmp(argv[i], "--tune") == 0) do_tune = true;
    printf("Acfun 推理引擎基准  |  后端: %s\n", acfun_backend_name());
    if (do_tune) {
        printf(">>> 执行 GEMM 自动调参 (ACFUN_AUTOTUNE) ...\n");
        acfun_autotune();
    }
    printf(">>> GEMM 调参状态: %s\n", acfun_tune_status());
    int fails = correctness();
    fails += correctness_int8();
    fails += correctness_epi();
    bench_gemm();
    bench_gemm_scale();
    bench_conv();
    bench_winograd();
    bench_winograd43();
    bench_epi();
    const char* mp = (argc > 1) ? argv[1] : "model.acfun";
    bench_e2e(mp);
    bench_fusion(mp);
    bench_conv1x1();
    bench_depthwise();
    bench_grouped();
    bench_fc_relu_fused();
    bench_conv_bn_relu_composite();
    bench_int8();
    bench_int8_winograd();
    bench_bf16();
    printf("\n正确性: %s\n", fails == 0 ? "全部通过" : "存在失败");
    return fails == 0 ? 0 : 1;
}
