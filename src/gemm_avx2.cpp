/* Acfun GEMM — AVX2 + FMA 后端 (GotoBLAS 风格三级分块)
 * 与 AVX-512 后端同构, 仅微内核为 8x8 (__m256, 256-bit 满宽)。采用同样的最优打包
 * (A 仅在 (m0,k0) 打一次, 复用于全部 n0)、N 外层/M 内层循环序, 按 m0 并行。
 * 编译单元需带 -mavx2 -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <unistd.h>

static int g_dim(const char* env, int def) {
    const char* e = std::getenv(env);
    if (e) { int v = std::atoi(e); if (v > 0) return v; }
    return def;
}

static void auto_blocks(int* KC, int* NC) {
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long cap = (l2 > 0) ? (l2 / 4 * 3) : (1L << 19);
    int kc = 256, nc = 256;
    while ((long)kc * nc * 4 + (long)256 * kc * 4 > cap && kc > 32) kc /= 2;
    while ((long)kc * nc * 4 + (long)256 * kc * 4 > cap && nc > 32) nc /= 2;
    *KC = kc; *NC = nc;
}

/* 微内核: 仅完整 8×8 块。循环序 N 外层 / M 内层 (B 复用)。 */
static inline void kernel_8x8(const float* Ap, const float* Bp,
                              float* C, int ldc,
                              int mc, int nc, int kc,
                              int MC, int NC, int beta) {
    int mrR = mc - (mc & 7);
    int nrR = nc - (nc & 7);
    for (int nn = 0; nn < nrR; nn += 8) {            /* N 外层: B 行被复用 */
        for (int mm = 0; mm < mrR; mm += 8) {       /* M 内层 */
            __m256 c[8];
            if (beta == 0) {
                for (int i = 0; i < 8; ++i) c[i] = _mm256_setzero_ps();
            } else {
                for (int i = 0; i < 8; ++i)
                    c[i] = _mm256_loadu_ps(C + ((size_t)(mm + i)) * ldc + nn);
            }
            for (int k = 0; k < kc; ++k) {
                __m256 b = _mm256_loadu_ps(Bp + ((size_t)k) * NC + nn);
                for (int i = 0; i < 8; ++i) {
                    float a = Ap[((size_t)k) * MC + mm + i];
                    c[i] = _mm256_fmadd_ps(_mm256_set1_ps(a), b, c[i]);
                }
            }
            for (int i = 0; i < 8; ++i)
                _mm256_storeu_ps(C + ((size_t)(mm + i)) * ldc + nn, c[i]);
        }
    }
}

static void gemm_mblock(int m0, int M, int N, int K,
                        const float* A, const float* B, float* C,
                        int MC, int NC, int KC,
                        float* Ap, float* Bp) {
    int mc = std::min(MC, M - m0);
    const int lda = K, ldb = N, ldc = N;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;

        for (int k = 0; k < kc; ++k)
            for (int i = 0; i < mc; ++i)
                Ap[((size_t)k) * MC + i] = A[((size_t)(m0 + i)) * lda + (k0 + k)];

        for (int n0 = 0; n0 < N; n0 += NC) {
            int nc = std::min(NC, N - n0);
            for (int k = 0; k < kc; ++k)
                std::memcpy(Bp + (size_t)k * NC,
                            B + ((size_t)(k0 + k)) * ldb + n0,
                            (size_t)nc * sizeof(float));

            int mrR = mc - (mc & 7), nrR = nc - (nc & 7);
            kernel_8x8(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc,
                       mc, nc, kc, MC, NC, beta);

            if (mc > mrR) {
                auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
                f(mc - mrR, nrR, kc, lda, ldb, ldc,
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
            if (mc > mrR && nc > nrR) {
                auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
                f(mc - mrR, nc - nrR, kc, lda, ldb, ldc,
                  A + ((size_t)(m0 + mrR)) * lda + k0,
                  B + ((size_t)k0) * ldb + n0 + nrR,
                  C + ((size_t)(m0 + mrR)) * ldc + n0 + nrR);
            }
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
    int nt = acfun_get_num_threads_runtime();
    int mb = (M + MC - 1) / MC;

    if (nt <= 1 || mb <= 1) {
        float* Ap = (float*)malloc((size_t)MC * KC * sizeof(float));
        float* Bp = (float*)malloc((size_t)KC * NC * sizeof(float));
        if (Ap && Bp) {
            for (int m0 = 0; m0 < M; m0 += MC)
                gemm_mblock(m0, M, N, K, A, B, C, MC, NC, KC, Ap, Bp);
        }
        free(Ap); free(Bp);
        return;
    }

    #pragma omp parallel num_threads(nt)
    {
        float* Ap = (float*)malloc((size_t)MC * KC * sizeof(float));
        float* Bp = (float*)malloc((size_t)KC * NC * sizeof(float));
        if (!Ap || !Bp) { free(Ap); free(Bp); goto end_par; }

        #pragma omp for schedule(static, 1)
        for (int m0b = 0; m0b < mb; ++m0b) {
            int m0 = m0b * MC;
            gemm_mblock(m0, M, N, K, A, B, C, MC, NC, KC, Ap, Bp);
        }
        end_par:
        free(Ap); free(Bp);
    }
}
