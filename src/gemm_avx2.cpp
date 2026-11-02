/* Acfun GEMM — AVX2 + FMA 后端 (GotoBLAS 风格三级分块)
 * 与 AVX-512 后端同构, 仅微内核为 8x8 (__m256, 256-bit 满宽)。采用同样的最优打包
 * (A 仅在 (m0,k0) 打一次, 复用于全部 n0)、N 外层/M 内层循环序, 按 m0 并行。
 * 编译单元需带 -mavx2 -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <vector>
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

static void auto_blocks(int MC, int* KC, int* NC) {
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long cap = (l2 > 0) ? (l2 / 2) : (1L << 19);
    int kc = 256, nc = 128;
    while ((long)kc * nc * 4 + (long)MC * kc * 4 > cap && kc > 32) kc /= 2;
    while ((long)kc * nc * 4 + (long)MC * kc * 4 > cap && nc > 32) nc /= 2;
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

/* 单个 (m0,n0) 矩形瓦片: 使用已打包好的 A 面板 (Apool_mid, 全 K 打包一次) 与本次
 * 打包的 B 面板, 完成该瓦片在全部 K 上的计算。每个瓦片独占其 C 矩形, β 跨 KC 块
 * 累加逻辑完全自洽、线程安全。 */
static void gemm_tile2(int mid, int n0, int M, int N, int K,
                       const float* A, const float* B, float* C,
                       int MC, int NC, int KC,
                       const float* Apool_mid, float* Bp) {
    int m0 = mid * MC;
    int mc = std::min(MC, M - m0);
    int nc = std::min(NC, N - n0);
    const int lda = K, ldb = N, ldc = N;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;   /* 同一瓦片跨 KC 块累加 */
        const float* Ap = Apool_mid + (size_t)k0 * MC;   /* 复用已打包 A */

        for (int k = 0; k < kc; ++k)
            std::memcpy(Bp + (size_t)k * NC,
                        B + ((size_t)(k0 + k)) * ldb + n0,
                        (size_t)nc * sizeof(float));

        int mrR = mc - (mc & 7), nrR = nc - (nc & 7);
        kernel_8x8(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc,
                   mc, nc, kc, MC, NC, beta);

        auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
        if (mc > mrR)
            f(mc - mrR, nrR, kc, lda, ldb, ldc,
              A + ((size_t)(m0 + mrR)) * lda + k0,
              B + ((size_t)k0) * ldb + n0,
              C + ((size_t)(m0 + mrR)) * ldc + n0);
        if (nc > nrR)
            f(mrR, nc - nrR, kc, lda, ldb, ldc,
              A + ((size_t)m0) * lda + k0,
              B + ((size_t)k0) * ldb + n0 + nrR,
              C + ((size_t)m0) * ldc + n0 + nrR);
        if (mc > mrR && nc > nrR)
            f(mc - mrR, nc - nrR, kc, lda, ldb, ldc,
              A + ((size_t)(m0 + mrR)) * lda + k0,
              B + ((size_t)k0) * ldb + n0 + nrR,
              C + ((size_t)(m0 + mrR)) * ldc + n0 + nrR);
    }
}

void acfun_gemm_avx2(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int MR = 8, NR = 8;
    int KC = 256, NC = 128;
    int MC = 256;
    int nt = acfun_get_num_threads_runtime();
    if (std::getenv("ACFUN_KC") || std::getenv("ACFUN_NC")) {
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 128);
    } else {
        /* 自适应 MC: 以 ~16 任务/线程为目标选取瓦片大小 (见 AVX-512 后端注释)。 */
        long target = (nt > 1) ? (long)16 * nt : 1;
        long mc_t = ((long)M * N) / (target * (long)NC);
        if (mc_t < 64) mc_t = 64;
        if (mc_t > 512) mc_t = 512;
        MC = (int)(((mc_t + MR - 1) / MR) * MR);
        auto_blocks(MC, &KC, &NC);
    }
    MC = (MC / MR) * MR; if (MC < MR) MC = MR;
    NC = (NC / NR) * NR; if (NC < NR) NC = NR;
    if (KC < 8) KC = 8;

    if (nt <= 1) {
        float* Ap = (float*)malloc((size_t)MC * KC * sizeof(float));
        float* Bp = (float*)malloc((size_t)KC * NC * sizeof(float));
        if (Ap && Bp) {
            for (int m0 = 0; m0 < M; m0 += MC)
                gemm_mblock(m0, M, N, K, A, B, C, MC, NC, KC, Ap, Bp);
        }
        free(Ap); free(Bp);
        return;
    }

    /* 内存带宽受限判定: A 矩阵单份即超过 L3, 走 m0 并行 (A 仅打一次、无 A 池额外流量);
     * 否则走 2D 瓦片并行 (缓存可容, 2D 并行收益显著)。 */
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    bool membound = (l3 > 0) && ((long)M * K * 4L > l3);

    if (membound) {
        int MCp = 128; MCp = (MCp / MR) * MR; if (MCp < MR) MCp = MR;
        int KCp = KC, NCp = NC; auto_blocks(MCp, &KCp, &NCp);
        int mbp = (M + MCp - 1) / MCp;
        std::vector<float*> bufA(nt, nullptr), bufB(nt, nullptr);
        for (int t = 0; t < nt; ++t) {
            bufA[t] = (float*)malloc((size_t)MCp * KCp * sizeof(float));
            bufB[t] = (float*)malloc((size_t)KCp * NCp * sizeof(float));
        }
        #pragma omp parallel num_threads(nt)
        {
            int tid = omp_get_thread_num();
            float* Ap = bufA[tid]; float* Bp = bufB[tid];
            if (!Ap || !Bp) goto end_p;
            #pragma omp for schedule(dynamic, 1)
            for (int m0b = 0; m0b < mbp; ++m0b)
                gemm_mblock(m0b * MCp, M, N, K, A, B, C, MCp, NCp, KCp, Ap, Bp);
            end_p:;
        }
        for (int t = 0; t < nt; ++t) { free(bufA[t]); free(bufB[t]); }
        return;
    }

    /* ---- 二维 (M×N) 瓦片并行 ---- */
    int mb = (M + MC - 1) / MC;
    int nb = (N + NC - 1) / NC;
    int nT = mb * nb;
    std::vector<float> Apool((size_t)mb * MC * K);

    std::vector<float*> bufB(nt, nullptr);
    for (int t = 0; t < nt; ++t)
        bufB[t] = (float*)malloc((size_t)KC * NC * sizeof(float));

    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num();
        float* Bp = bufB[tid];
        if (!Bp) goto end_par;

        /* Phase 1 — 并行打包各 m0 宏块的 A 面板 (每个仅一次) */
        #pragma omp for schedule(dynamic, 1) nowait
        for (int mid = 0; mid < mb; ++mid) {
            int m0 = mid * MC;
            int mc = std::min(MC, M - m0);
            float* Ap = Apool.data() + (size_t)mid * MC * K;
            const int lda = K;
            for (int k = 0; k < K; ++k)
                for (int i = 0; i < mc; ++i)
                    Ap[(size_t)k * MC + i] = A[((size_t)(m0 + i)) * lda + k];
        }
        #pragma omp barrier

        /* Phase 2 — 并行计算 2D 瓦片网格 (共享只读 A 面板) */
        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < nT; ++t) {
            int mid = t / nb;
            int n0 = (t % nb) * NC;
            gemm_tile2(mid, n0, M, N, K, A, B, C, MC, NC, KC,
                       Apool.data() + (size_t)mid * MC * K, Bp);
        }
        end_par:;
    }

    for (int t = 0; t < nt; ++t) free(bufB[t]);
}
