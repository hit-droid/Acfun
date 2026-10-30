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

int main(int argc, char** argv) {
    printf("Acfun 推理引擎基准  |  后端: %s\n", acfun_backend_name());
    int fails = correctness();
    bench_gemm();
    bench_conv();
    const char* mp = (argc > 1) ? argv[1] : "model.acfun";
    bench_e2e(mp);
    printf("\n正确性: %s\n", fails == 0 ? "全部通过" : "存在失败");
    return fails == 0 ? 0 : 1;
}
