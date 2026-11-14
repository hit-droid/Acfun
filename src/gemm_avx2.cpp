/* Acfun GEMM — AVX2 + FMA 后端 (GotoBLAS 风格三级分块)
 *
 * 与 AVX-512 后端同构, 仅微内核为 MR=6 × NR=16 (__m256, 256-bit 满宽):
 *   每个 k 迭代 6 次 A 广播 + 12 次 FMA (6 行 × 2 路 8 列), port5(广播) 与
 *   FMA 端口 (0/1, 2/cycle) 均衡 → 单核逼近 AVX2 FMA 峰值。
 *   寄存器预算: 12 个 c 累加器 (__m256) + 2 个 b 向量 = 14 ≤ 16 ymm。
 *   同样采用 A、B 各仅全 K 打包一次的 2D 瓦片并行, 保证多核随核数近线性扩展。
 *   融合后处理 (epilogue fusion) 与 AVX-512 后端同构: 在写回 C 的同一最内层循环里
 *   就地叠加 bias[j] + 可选 ReLU, 仅在末 KC 块进行, 省去一层全张量扫描; bias=NULL
 *   且 act=0 时该分支不进入, 与无融合路径逐指令一致、零开销。
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

/* 融合后处理: 在 C 矩形 [r0,r1)×[c0,c1) 上就地叠 bias[j] + 可选 ReLU (末 KC 块调用)。 */
static inline void apply_epi(float* C, int r0, int r1, int c0, int c1, int N,
                             const float* bias, int act) {
    if (!bias && !act) return;
    for (int i = r0; i < r1; ++i) {
        float* p = C + (size_t)i * N + c0;
        for (int j = c0; j < c1; ++j) {
            float v = p[j - c0];
            if (bias) v += bias[j];
            if (act && v < 0.0f) v = 0.0f;
            p[j - c0] = v;
        }
    }
}

/* 微内核: 仅处理完整 MR×NR 块。Cbase 为瓦片在 C 中的原点 (m0*ldc+n0)。
 * 末 KC 块 (last) 写回时, 对每个 16 列块载入 bias[n0+nn..+15] 一次, 12 路累加器
 * (两路 8 列) 各 +bias 后 (可选) 取 max(0,·); 非末块恒不叠加。 */
static inline void kernel(const float* Ap, const float* Bp, float* Cbase, int ldc,
                          int mc, int nc, int kc, int MC, int NC, int beta,
                          const float* bias, int n0, int act, int last) {
    int mrR = mc - (mc % MR);
    int nrR = nc - (nc % NR);
    __m256 z = _mm256_setzero_ps();
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
            /* ---- 融合后处理: 仅末 KC 块 ---- */
            if (last) {
                __m256 b0v = bias ? _mm256_loadu_ps(bias + n0 + nn)     : z;
                __m256 b1v = bias ? _mm256_loadu_ps(bias + n0 + nn + 8) : z;
                for (int i = 0; i < MR; ++i) {
                    if (bias) { c0[i] = _mm256_add_ps(c0[i], b0v); c1[i] = _mm256_add_ps(c1[i], b1v); }
                    if (act)   { c0[i] = _mm256_max_ps(c0[i], z);    c1[i] = _mm256_max_ps(c1[i], z); }
                }
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
                              int MC, int NC, int kc, int mc, int nc, int k0, int beta,
                              const float* bias, int act, int last) {
    const int lda = K, ldb = N, ldc = N;
    int mrR = mc - (mc % MR);
    int nrR = nc - (nc % NR);
    auto f = beta ? acfun_gemm_ref_ld_b1 : acfun_gemm_ref_ld;
    if (mc > mrR) {
        int r0 = m0 + mrR, r1 = m0 + mc, c0 = n0, c1 = n0 + nrR;
        f(mc - mrR, nrR, kc, lda, ldb, ldc,
          A + ((size_t)(m0 + mrR)) * lda + k0, B + (size_t)k0 * ldb + n0,
          C + ((size_t)r0) * ldc + c0);
        if (last) apply_epi(C, r0, r1, c0, c1, N, bias, act);
    }
    if (nc > nrR) {
        int r0 = m0, r1 = m0 + mrR, c0 = n0 + nrR, c1 = n0 + nc;
        f(mrR, nc - nrR, kc, lda, ldb, ldc,
          A + (size_t)m0 * lda + k0, B + (size_t)k0 * ldb + n0 + nrR,
          C + ((size_t)r0) * ldc + c0);
        if (last) apply_epi(C, r0, r1, c0, c1, N, bias, act);
    }
    if (mc > mrR && nc > nrR) {
        int r0 = m0 + mrR, r1 = m0 + mc, c0 = n0 + nrR, c1 = n0 + nc;
        f(mc - mrR, nc - nrR, kc, lda, ldb, ldc,
          A + ((size_t)(m0 + mrR)) * lda + k0, B + (size_t)k0 * ldb + n0 + nrR,
          C + ((size_t)r0) * ldc + c0);
        if (last) apply_epi(C, r0, r1, c0, c1, N, bias, act);
    }
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
                             const float* Ap_mid, const float* Bp_nid,
                             const float* bias, int act) {
    int mc = std::min(MC, M - m0);
    int nc = std::min(NC, N - n0);
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;
        int last = (k0 + kc >= K) ? 1 : 0;
        kernel(Ap_mid + (size_t)k0 * MC, Bp_nid + (size_t)k0 * NC,
               C + (size_t)m0 * N + n0, N, mc, nc, kc, MC, NC, beta,
               bias, n0, act, last);
        tile_tails(m0, n0, M, N, K, A, B, C, MC, NC, kc, mc, nc, k0, beta,
                   bias, act, last);
    }
}

static void gemm_mblock_stream(int m0, int M, int N, int K,
                               const float* A, const float* B, float* C,
                               int MC, int NC, int KC, float* Ap, float* Bp,
                               const float* bias, int act) {
    int mc = std::min(MC, M - m0);
    const int lda = K, ldb = N, ldc = N;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;
        int last = (k0 + kc >= K) ? 1 : 0;

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

            kernel(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc, mc, nc, kc, MC, NC, beta,
                   bias, n0, act, last);
            tile_tails(m0, n0, M, N, K, A, B, C, MC, NC, kc, mc, nc, k0, beta,
                       bias, act, last);
        }
    }
}

static void gemm_avx2_core(int M, int N, int K,
                           const float* A, const float* B, float* C,
                           const float* bias, int act) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int KC = 256, NC = 128, MC = 256;
    int nt = acfun_get_num_threads_runtime();
    int tkc = 0, tnc = 0;
    int have_tune = acfun_get_tuned_kc_nc(nt, &tkc, &tnc);
    int env_kc = (std::getenv("ACFUN_KC") != nullptr);
    int env_nc = (std::getenv("ACFUN_NC") != nullptr);
    if (env_kc || env_nc) {
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 128);
        MC = 256;
    } else if (have_tune) {
        KC = tkc; NC = tnc;
        long target = (nt > 1) ? (long)16 * nt : 1;
        long mc_t = ((long)M * N) / (target * (long)NC);
        if (mc_t < 64) mc_t = 64;
        if (mc_t > 512) mc_t = 512;
        MC = (int)(((mc_t + MR - 1) / MR) * MR);
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
                gemm_mblock_stream(m0, M, N, K, A, B, C, MC, NC, KC, Ap, Bp, bias, act);
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
                             Bpool + (size_t)nid * NC * K, bias, act);
        }
    }
    free(Apool); free(Bpool);
}

void acfun_gemm_avx2(int M, int N, int K,
                     const float* A, const float* B, float* C) {
    gemm_avx2_core(M, N, K, A, B, C, nullptr, 0);
}

void acfun_gemm_avx2_epi(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         const float* bias, int act) {
    gemm_avx2_core(M, N, K, A, B, C, bias, act);
}
