/* Acfun GEMM — AVX2 + FMA 后端 (GotoBLAS 风格三级分块)
 *
 * 与 AVX-512 后端同构, 仅微内核为 MR=6 × NR=16 (__m256, 256-bit 满宽):
 *   每个 k 迭代 6 次 A 广播 + 12 次 FMA (6 行 × 2 路 8 列), port5(广播) 与
 *   FMA 端口 (0/1, 2/cycle) 均衡 → 单核逼近 AVX2 FMA 峰值。
 *   寄存器预算: 12 个 c 累加器 (__m256) + 2 个 b 向量 = 14 ≤ 16 ymm。
 *   同样采用 A、B 各仅全 K 打包一次的 2D 瓦片并行, 保证多核随核数近线性扩展。
 *   编译单元需带 -mavx2 -mfma。 */
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

/* 微内核尺寸: 6 行 × 16 列 (两路 8 列)。 */
static const int MR = 6;
static const int NR = 16;

/* 微内核: 仅处理完整 MR×NR 块。Cbase 为瓦片在 C 中的原点 (m0*ldc+n0)。 */
static inline void kernel(const float* Ap, const float* Bp, float* Cbase, int ldc,
                          int mc, int nc, int kc, int MC, int NC, int beta) {
    int mrR = mc - (mc % MR);
    int nrR = nc - (nc % NR);
    for (int nn = 0; nn < nrR; nn += NR) {          /* N 外层: B 行被复用 */
        for (int mm = 0; mm < mrR; mm += MR) {      /* M 内层: 6 行 */
            __m256 c0[MR], c1[MR];                  /* 两路 8 列累加器 */
            float* cp = Cbase + (size_t)mm * ldc + nn;
            if (beta == 0) {
                for (int i = 0; i < MR; ++i) {
                    c0[i] = _mm256_setzero_ps();
                    c1[i] = _mm256_setzero_ps();
                }
            } else {
                for (int i = 0; i < MR; ++i) {
                    c0[i] = _mm256_loadu_ps(cp + (size_t)i * ldc);
                    c1[i] = _mm256_loadu_ps(cp + (size_t)i * ldc + 8);
                }
            }
            for (int k = 0; k < kc; ++k) {
                const float* bp = Bp + (size_t)k * NC + nn;
                __m256 b0 = _mm256_loadu_ps(bp);
                __m256 b1 = _mm256_loadu_ps(bp + 8);
                const float* ap = Ap + (size_t)k * MC + mm;
                for (int i = 0; i < MR; ++i) {
                    __m256 a = _mm256_broadcastss_ps(_mm_load_ss(ap + i));
                    c0[i] = _mm256_fmadd_ps(a, b0, c0[i]);
                    c1[i] = _mm256_fmadd_ps(a, b1, c1[i]);
                }
                _mm_prefetch((const char*)(Bp + (size_t)(k + 4) * NC + nn), _MM_HINT_T0);
                _mm_prefetch((const char*)(Ap + (size_t)(k + 4) * MC + mm), _MM_HINT_T0);
            }
            for (int i = 0; i < MR; ++i) {
                _mm256_storeu_ps(cp + (size_t)i * ldc, c0[i]);
                _mm256_storeu_ps(cp + (size_t)i * ldc + 8, c1[i]);
            }
        }
    }
}

static inline void tile_tails(int m0, int n0, int M, int N, int K,
                              const float* A, const float* B, float* C,
                              int MC, int NC, int kc, int mc, int nc, int k0, int beta) {
    const int lda = K, ldb = N, ldc = N;
    int mrR = mc - (mc % MR);
    int nrR = nc - (nc % NR);
    auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
    if (mc > mrR)
        f(mc - mrR, nrR, kc, lda, ldb, ldc,
          A + ((size_t)(m0 + mrR)) * lda + k0, B + (size_t)k0 * ldb + n0,
          C + ((size_t)(m0 + mrR)) * ldc + n0);
    if (nc > nrR)
        f(mrR, nc - nrR, kc, lda, ldb, ldc,
          A + (size_t)m0 * lda + k0, B + (size_t)k0 * ldb + n0 + nrR,
          C + (size_t)m0 * ldc + (n0 + nrR));
    if (mc > mrR && nc > nrR)
        f(mc - mrR, nc - nrR, kc, lda, ldb, ldc,
          A + ((size_t)(m0 + mrR)) * lda + k0, B + (size_t)k0 * ldb + n0 + nrR,
          C + ((size_t)(m0 + mrR)) * ldc + (n0 + nrR));
}

static inline void packA_all(float* Ap, const float* A, int m0, int mc, int K, int MC) {
    const int lda = K;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < mc; ++i)
            Ap[(size_t)k * MC + i] = A[((size_t)(m0 + i)) * lda + k];
}

static inline void packB_all(float* Bp, const float* B, int n0, int nc, int K, int NC, int N) {
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < nc; ++j)
            Bp[(size_t)k * NC + j] = B[((size_t)k) * N + n0 + j];
}

static void gemm_tile_packed(int m0, int n0, int M, int N, int K,
                             const float* A, const float* B, float* C,
                             int MC, int NC, int KC,
                             const float* Ap_mid, const float* Bp_nid) {
    int mc = std::min(MC, M - m0);
    int nc = std::min(NC, N - n0);
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;
        kernel(Ap_mid + (size_t)k0 * MC, Bp_nid + (size_t)k0 * NC,
               C + (size_t)m0 * N + n0, N, mc, nc, kc, MC, NC, beta);
        tile_tails(m0, n0, M, N, K, A, B, C, MC, NC, kc, mc, nc, k0, beta);
    }
}

static void gemm_mblock_stream(int m0, int M, int N, int K,
                               const float* A, const float* B, float* C,
                               int MC, int NC, int KC, float* Ap, float* Bp) {
    int mc = std::min(MC, M - m0);
    const int lda = K, ldb = N, ldc = N;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;

        /* 打包 A 面板: kk 外 / i 内 → Ap[kk*MC + i] 连续写入, 可被向量化;
         * i 外 / kk 内则跨 MC 步长散写, 窄 N 时成为瓶颈。 */
        for (int kk = 0; kk < kc; ++kk)
            for (int i = 0; i < mc; ++i)
                Ap[(size_t)kk * MC + i] = A[((size_t)(m0 + i)) * lda + (k0 + kk)];

        for (int n0 = 0; n0 < N; n0 += NC) {
            int nc = std::min(NC, N - n0);
            for (int kk = 0; kk < kc; ++kk)
                for (int j = 0; j < nc; ++j)
                    Bp[(size_t)kk * NC + j] = B[((size_t)(k0 + kk)) * ldb + n0 + j];

            kernel(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc, mc, nc, kc, MC, NC, beta);
            tile_tails(m0, n0, M, N, K, A, B, C, MC, NC, kc, mc, nc, k0, beta);
        }
    }
}

void acfun_gemm_avx2(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int KC = 256, NC = 128, MC = 256;
    int nt = acfun_get_num_threads_runtime();
    if (std::getenv("ACFUN_KC") || std::getenv("ACFUN_NC")) {
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 128);
    } else {
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
                gemm_mblock_stream(m0, M, N, K, A, B, C, MC, NC, KC, Ap, Bp);
        }
        free(Ap); free(Bp);
        return;
    }

    int mb = (M + MC - 1) / MC;
    int nb = (N + NC - 1) / NC;
    float* Apool = (float*)malloc((size_t)mb * MC * K * sizeof(float));
    float* Bpool = (float*)malloc((size_t)nb * NC * K * sizeof(float));
    if (!Apool || !Bpool) { free(Apool); free(Bpool); return; }

    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for schedule(dynamic, 1) nowait
        for (int mid = 0; mid < mb; ++mid) {
            int m0 = mid * MC;
            int mc = std::min(MC, M - m0);
            packA_all(Apool + (size_t)mid * MC * K, A, m0, mc, K, MC);
        }
        #pragma omp for schedule(dynamic, 1) nowait
        for (int nid = 0; nid < nb; ++nid) {
            int n0 = nid * NC;
            int nc = std::min(NC, N - n0);
            packB_all(Bpool + (size_t)nid * NC * K, B, n0, nc, K, NC, N);
        }
        #pragma omp barrier

        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < mb * nb; ++t) {
            int mid = t / nb;
            int nid = t % nb;
            int m0 = mid * MC;
            int n0 = nid * NC;
            gemm_tile_packed(m0, n0, M, N, K, A, B, C, MC, NC, KC,
                             Apool + (size_t)mid * MC * K,
                             Bpool + (size_t)nid * NC * K);
        }
    }
    free(Apool); free(Bpool);
}
