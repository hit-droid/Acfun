/* Acfun GEMM — AVX2 + FMA 后端 (GotoBLAS 风格三级分块)
 * 与 AVX-512 后端同构, 仅微内核为 8x8 (__m256, 256-bit 满宽)。
 * 编译单元需带 -mavx2 -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <unistd.h>

static int g_dim(const char* env, int def) {
    const char* e = std::getenv(env);
    if (e) { int v = std::atoi(e); if (v > 0) return v; }
    return def;
}

static void auto_blocks(int* KC, int* NC) {
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long cap = (l2 > 0) ? l2 / 2 : (1L << 19);
    int kc = 256, nc = 256;
    if ((long)kc * nc * 4 + (long)256 * kc * 4 > cap) { kc = 128; nc = 128; }
    if ((long)kc * nc * 4 + (long)256 * kc * 4 > cap) { kc = 64;  nc = 64;  }
    *KC = kc; *NC = nc;
}

static inline void kernel_8x8(const float* Ap, const float* Bp,
                              float* C, int ldC,
                              int mc, int nc, int kc,
                              int MC, int NC, int beta) {
    int mrR = mc - (mc & 7);
    int nrR = nc - (nc & 7);
    for (int mm = 0; mm < mrR; mm += 8) {
        for (int nn = 0; nn < nrR; nn += 8) {
            __m256 c[8];
            if (beta == 0) {
                for (int i = 0; i < 8; ++i) c[i] = _mm256_setzero_ps();
            } else {
                for (int i = 0; i < 8; ++i)
                    c[i] = _mm256_loadu_ps(C + ((size_t)(mm + i)) * ldC + nn);
            }
            for (int k = 0; k < kc; ++k) {
                __m256 b = _mm256_loadu_ps(Bp + ((size_t)k) * NC + nn);
                for (int i = 0; i < 8; ++i) {
                    float a = Ap[((size_t)k) * MC + mm + i];
                    c[i] = _mm256_fmadd_ps(_mm256_set1_ps(a), b, c[i]);
                }
                _mm_prefetch((const char*)(Bp + ((size_t)(k + 16)) * NC + nn), _MM_HINT_T0);
                _mm_prefetch((const char*)(Ap + ((size_t)(k + 16)) * MC + mm), _MM_HINT_T0);
            }
            for (int i = 0; i < 8; ++i)
                _mm256_storeu_ps(C + ((size_t)(mm + i)) * ldC + nn, c[i]);
        }
    }
}

void acfun_gemm_avx2(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int MR = 8, NR = 8;
    int MC = 256;
    int KC = 256, NC = 256;
    if (std::getenv("ACFUN_KC") || std::getenv("ACFUN_NC")) {
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 256);
    } else {
        auto_blocks(&KC, &NC);
    }
    MC = (MC / MR) * MR; if (MC < MR) MC = MR;
    NC = (NC / NR) * NR; if (NC < NR) NC = NR;
    if (KC < 8) KC = 8;
    const int lda = K, ldb = N, ldc = N;

    #pragma omp parallel
    {
        float* Ap = (float*)malloc((size_t)MC * KC * sizeof(float));
        float* Bp = (float*)malloc((size_t)KC * NC * sizeof(float));
        if (!Ap || !Bp) { free(Ap); free(Bp); goto end_par; }

        #pragma omp for schedule(static, 1)
        for (int n0 = 0; n0 < N; n0 += NC) {
            int nc = std::min(NC, N - n0);
            for (int k0 = 0; k0 < K; k0 += KC) {
                int kc = std::min(KC, K - k0);
                int beta = (k0 == 0) ? 0 : 1;

                for (int k = 0; k < kc; ++k)
                    std::memcpy(Bp + (size_t)k * NC,
                                B + ((size_t)(k0 + k)) * ldb + n0,
                                (size_t)nc * sizeof(float));

                for (int m0 = 0; m0 < M; m0 += MC) {
                    int mc = std::min(MC, M - m0);
                    for (int k = 0; k < kc; ++k)
                        for (int i = 0; i < mc; ++i)
                            Ap[((size_t)k) * MC + i] = A[((size_t)(m0 + i)) * lda + (k0 + k)];

                    kernel_8x8(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc,
                               mc, nc, kc, MC, NC, beta);

                    int mrR = mc - (mc & 7), nrR = nc - (nc & 7);
                    if (mc > mrR) {
                        auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
                        f(mc - mrR, nc, kc, lda, ldb, ldc,
                          A + ((size_t)(m0 + mrR)) * lda + k0,
                          B + ((size_t)k0) * ldb + n0,
                          C + ((size_t)(m0 + mrR)) * ldc + n0);
                    }
                    if (nc > nrR) {
                        auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
                        f(mrR, nc - nrR, kc, lda, ldb, ldc,
                          A + ((size_t)m0) * lda + k0,
                          B + ((size_t)k0) * ldb + n0 + nrR,
                          C + ((size_t)m0) * ldc + n0 + nrR);
                    }
                }
            }
        }
        end_par:
        free(Ap); free(Bp);
    }
}
