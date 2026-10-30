/* Acfun GEMM — AVX2 + FMA 后端
 * 微内核 8x8 (256-bit 满宽): C 块 64 累加器驻留 ymm 跨整个 K 累加,
 * A 面板按 k 连续重排命中 L2, B 直接访问, OpenMP 并行 M 维。
 * 编译单元需带 -mavx2 -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>

void acfun_gemm_avx2(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int MR = 8, NR = 8;
    const int MC = 32;

    #pragma omp parallel
    {
        float* Ap = (float*)malloc((size_t)MC * K * sizeof(float));
        if (!Ap) goto end_par;

        #pragma omp for schedule(static)
        for (int m0 = 0; m0 < M; m0 += MC) {
            int MCe = std::min(MC, M - m0);
            int mrR = MCe - (MCe % MR);
            int ncR = N - (N % NR);

            for (int k = 0; k < K; ++k)
                for (int i = 0; i < MCe; ++i)
                    Ap[(size_t)k * MCe + i] = A[((size_t)(m0 + i)) * K + k];

            for (int nn = 0; nn + NR <= N; nn += NR) {
                for (int mm = 0; mm < mrR; mm += MR) {
                    __m256 c[MR];
                    for (int i = 0; i < MR; ++i) c[i] = _mm256_setzero_ps();
                    for (int k = 0; k < K; ++k) {
                        __m256 b = _mm256_loadu_ps(B + ((size_t)k) * N + nn);
                        for (int i = 0; i < MR; ++i) {
                            float a = Ap[(size_t)k * MCe + mm + i];
                            c[i] = _mm256_fmadd_ps(_mm256_set1_ps(a), b, c[i]);
                        }
                    }
                    for (int i = 0; i < MR; ++i)
                        _mm256_storeu_ps(C + ((size_t)(m0 + mm + i)) * N + nn, c[i]);
                }
            }
            if (MCe > mrR)
                acfun_gemm_ref_ld(MCe - mrR, N, K, K, N, N,
                                  A + (size_t)(m0 + mrR) * K, B,
                                  C + (size_t)(m0 + mrR) * N);
            if (N > ncR)
                acfun_gemm_ref_ld(mrR, N - ncR, K, K, N, N,
                                  A + (size_t)m0 * K, B + ncR,
                                  C + (size_t)m0 * N + ncR);
        }
        end_par:
        free(Ap);
    }
}
