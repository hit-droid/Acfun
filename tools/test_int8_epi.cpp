/* INT8 GEMM epilogue 融合验证: acfun_gemm_int8_epi (融合写回) vs
 * 「acfun_gemm_int8 写 int32 + 独立一遍反量化(同 alpha/beta)+bias+ReLU」参考。
 * 覆盖方阵/非方阵、非整除边界 (含 M/N/K 非 4/16 倍数, 触发微内核尾部块与 gather 钳位),
 * 以及有/无 ReLU 组合。二者用完全相同的 alpha/beta 公式, 应逐元素严格 max|Δ|=0。
 * 同时交叉验证标量参考 acfun_gemm_int8_epi_ref 与 SIMD 后端一致。
 *
 * 用法:
 *   g++ -O3 -std=c++17 -fopenmp -Iinclude -Iinclude/acfun \
 *       tools/test_int8_epi.cpp build/libacfun.a -o /tmp/test_int8_epi && /tmp/test_int8_epi */
#include "acfun.h"
#include "internal.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static void rf(std::vector<uint8_t>& v, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& x : v) x = (uint8_t)d(g);
}
static void rf(std::vector<int8_t>& v, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_int_distribution<int> d(-128, 127);
    for (auto& x : v) x = (int8_t)d(g);
}

/* 非融合参考: GEMM→int32, 再独立一遍反量化, 用与融合完全相同的 alpha/beta */
static void ref_epi(int M, int N, int K,
                    const uint8_t* A, const int8_t* B, float* C,
                    const float* alpha, const float* beta, int act) {
    std::vector<int32_t> i32((size_t)M * N);
    acfun_gemm_int8(M, N, K, A, B, i32.data());
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float v = (float)i32[(size_t)i * N + j] * alpha[j] + beta[j];
            if (act && v < 0.0f) v = 0.0f;
            C[(size_t)i * N + j] = v;
        }
}

int main() {
    printf("=== INT8 GEMM epilogue 融合正确性 ===\n");
    struct Cfg { int M, N, K; } cfgs[] = {
        {64,64,64}, {100,100,100}, {255,255,255}, {256,256,256},
        {127,129,63}, {16,16,16}, {1,16,255}, {513,17,131},
        {4,4,4}, {33,65,129}, {1023,4,7}, {7,1023,5},
    };
    acfun_set_num_threads_runtime(1);   /* 单线程先严格比对 */
    int fails = 0, cases = 0;
    unsigned seed = 12345;
    for (auto& c : cfgs) {
        for (int act = 0; act <= 1; ++act) {
            std::vector<uint8_t> A((size_t)c.M * c.K);
            std::vector<int8_t>  B((size_t)c.K * c.N);
            rf(A, seed++); rf(B, seed++);
            std::vector<float> alpha(c.N), beta(c.N);
            for (int j = 0; j < c.N; ++j) {
                alpha[j] = 0.001f + 0.0005f * (float)(j % 13);
                beta[j]  = -0.3f + 0.02f * (float)(j % 7);
            }
            std::vector<float> cref((size_t)c.M * c.N), csim((size_t)c.M * c.N), cref2((size_t)c.M * c.N);
            ref_epi(c.M, c.N, c.K,    A.data(), B.data(), cref.data(),  alpha.data(), beta.data(), act);
            acfun_gemm_int8_epi(c.M, c.N, c.K, A.data(), B.data(), csim.data(), alpha.data(), beta.data(), act);
            acfun_gemm_int8_epi_ref(c.M, c.N, c.K, A.data(), B.data(), cref2.data(), alpha.data(), beta.data(), act);
            float maxd = 0.0f, maxd2 = 0.0f;
            for (size_t i = 0; i < cref.size(); ++i) {
                maxd  = std::max(maxd,  std::fabs(cref[i]  - csim[i]));
                maxd2 = std::max(maxd2, std::fabs(cref2[i] - csim[i]));
            }
            bool ok = (maxd == 0.0f) && (maxd2 == 0.0f);
            printf("  %4dx%4dx%4d act=%d  SIMD vs 非融合 max|Δ|=%.3e  ref vs SIMD max|Δ|=%.3e  %s\n",
                   c.M, c.N, c.K, act, maxd, maxd2, ok ? "OK" : "FAIL");
            if (!ok) ++fails;
            ++cases;
        }
    }
    /* 多线程一致性: 4 线程结果须与单线程(非融合参考)一致 */
    acfun_set_num_threads_runtime(4);
    {
        int M=512,N=512,K=512, act=1;
        std::vector<uint8_t> A((size_t)M*K); std::vector<int8_t> B((size_t)K*N);
        rf(A, seed++); rf(B, seed++);
        std::vector<float> alpha(N), beta(N);
        for (int j=0;j<N;++j){ alpha[j]=0.002f; beta[j]=0.01f*(float)(j%5); }
        std::vector<float> r((size_t)M*N), o((size_t)M*N);
        ref_epi(M,N,K,A.data(),B.data(),r.data(),alpha.data(),beta.data(),act);
        acfun_gemm_int8_epi(M,N,K,A.data(),B.data(),o.data(),alpha.data(),beta.data(),act);
        float maxd=0; for(size_t i=0;i<r.size();++i) maxd=std::max(maxd,std::fabs(r[i]-o[i]));
        printf("  %4dx%4dx%4d act=%d  MT(4T) vs 非融合 max|Δ|=%.3e  %s\n", M,N,K,act,maxd,maxd==0.0f?"OK":"FAIL");
        if (maxd != 0.0f) ++fails;
        ++cases;
    }
    printf("结果: %d/%d 通过, FAIL=%d\n", cases - fails, cases, fails);
    return fails ? 1 : 0;
}
