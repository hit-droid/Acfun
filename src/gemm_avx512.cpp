/* Acfun GEMM — AVX-512F + FMA 后端
 * 微内核 16x16: 每个 C 块 256 个累加器驻留 zmm 寄存器跨整个 K 累加,
 * 算术强度高; A 面板按 k 连续重排到 L2, B 直接以 k 跨步访问(单块工作集命中 L2),
 * OpenMP 并行 M 维(细粒度分块)以充分利用多核。编译单元需带 -mavx512f -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>

void acfun_gemm_avx512(int M, int N, int K,
                       const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int MR = 16, NR = 16;
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

            /* 打包 A 面板: Ap[k*MCe + i] = A[(m0+i)*K + k] (k 连续, 命中 L2) */
            for (int k = 0; k < K; ++k)
                for (int i = 0; i < MCe; ++i)
                    Ap[(size_t)k * MCe + i] = A[((size_t)(m0 + i)) * K + k];

            for (int nn = 0; nn + NR <= N; nn += NR) {
                for (int mm = 0; mm < mrR; mm += MR) {
                    __m512 c[MR];
                    for (int i = 0; i < MR; ++i) c[i] = _mm512_setzero_ps();
                    for (int k = 0; k < K; ++k) {
                        __m512 b = _mm512_loadu_ps(B + ((size_t)k) * N + nn);
                        for (int i = 0; i < MR; ++i) {
                            float a = Ap[(size_t)k * MCe + mm + i];
                            c[i] = _mm512_fmadd_ps(_mm512_set1_ps(a), b, c[i]);
                        }
                    }
                    for (int i = 0; i < MR; ++i)
                        _mm512_storeu_ps(C + ((size_t)(m0 + mm + i)) * N + nn, c[i]);
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
