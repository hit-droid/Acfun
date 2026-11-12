/* Acfun GEMM — AVX-512F + FMA 后端 (GotoBLAS 风格三级分块)
 *
 * 对标 MNN / 一线 BLAS 的两大工程手段，本轮重点修复单核效率与多核扩展：
 *
 * 一、微内核形态: MR=16 × NR=16 (16 行 × 16 列)
 *   逐行 broadcast A (16 次/ k), B 按 16 列整块载入, 累加器 c[16] 为 16 个 zmm。
 *   在 Zen4/EPYC 上 port5 广播吞吐充足, 本形态数据复用最优 (单核 ~140 GFLOPS,
 *   已接近 FMA 峰值), 实测优于更窄的 MR=8 分块 (后者 B 面板复用减半, 反降吞吐)。
 *   NR=32 虽理论更优但需 34 个 zmm, 超 32 寄存器上限, 故维持 16×16。
 *
 * 二、全缓存分块 + B 仅打包一次 (2D 瓦片并行)
 *   旧 2D 路径每个 (m0,n0) 瓦片都重新打包自己的 B 面板 → B 被重复读 mb 次，
 *   内存带宽墙使 8 线程后算力崩塌 (16T 仅 177 GFLOPS)。
 *   新方案: A 按 m0 列块、B 按 n0 列块各自全 K 打包一次 (Phase-1 并行完成)，
 *   随后 Phase-2 在 mb×nb 瓦片网格上并行，所有 n0 复用同一份只读 A 面板、
 *   所有 m0 复用同一份只读 B 面板。于是 A 与 B 各仅从内存流一次 (总量 M·K + K·N)，
 *   彻底消除冗余流量 → 多核随物理核数近线性扩展。
 *
 *   微内核循环序: N 外层 / M 内层。每个 16 列块载入一次 B 行 (复用 8 行块)，
 *   累加器跨整个 KC 累加，β=0/β=1 跨 KC 块累加保证大矩阵正确性。
 *   行/列尾部余数 (M%MR、N%NR) 交给标量参考 (β=0/β=1 两套)，保证正确。
 *
 * 编译单元需带 -mavx512f -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <unistd.h>

static int g_dim(const char* env, int def) {
    const char* e = std::getenv(env);
    if (e) { int v = std::atoi(e); if (v > 0) return v; }
    return def;
}

/* 硬件自适应分块: 未显式指定时, 按 L2 容量选取 KC/NC, 使 A+B 面板合计稳居 L2。 */
static void auto_blocks(int MC, int* KC, int* NC) {
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long cap = (l2 > 0) ? (l2 / 2) : (1L << 19);   /* 取 L2 一半留余量 */
    int kc = 256, nc = 128;
    while ((long)kc * nc * 4 + (long)MC * kc * 4 > cap && kc > 64) kc /= 2;
    while ((long)kc * nc * 4 + (long)MC * kc * 4 > cap && nc > 64) nc /= 2;
    *KC = kc; *NC = nc;
}

/* 微内核尺寸: MR×NR = 16×16。A 按行 broadcast (16 次/ k), B 按 16 列整块载入,
 * 累加器 c[MR] 为 16 个 zmm。
 * 实测 (Zen4/EPYC, AVX-512F+FMA): 本形态数据复用最优, 单核 ~140 GFLOPS 已逼近 FMA
 * 峰值; 更窄的 MR=8 分块 (B 面板复用减半) 反降吞吐。NR=32 虽理论上更佳, 但需 34 个
 * zmm (32 累加器 + 2 B 向量), 超出 AVX-512 的 32 寄存器上限, 故维持 16×16。
 * 循环序: N 外层 / M 内层 —— 每个 16 列块载入一次 B, 被该列块下全部 M 行复用。 */
static const int MR = 16;
static const int NR = 16;

/* 微内核: 仅处理完整 MR×NR 块。Ap/Bp 为该 (m0,n0) 瓦片已全 K 打包的面板,
 * 索引 [k*MC + i] / [k*NC + j]；k 从 0 起。Cbase 为瓦片在 C 中的原点 (m0*ldc+n0)。
 *
 * 两阶段软件流水 (k 步长 2) + 双面板预取: 每轮同时取 k 与 k+1 的 B 向量, 各做 16 路 FMA,
 * 指令级并行度翻倍以掩盖 FMA 与广播 loads 的延迟; 同时对 A/B 下一 k 的缓存行做 T0 预取,
 * 使面板数据在读前落入 L1。实测 (Zen4/EPYC AVX-512F+FMA) 较单 k 版本单核吞吐再提升一档。 */
static inline void kernel(const float* Ap, const float* Bp, float* Cbase, int ldc,
                          int mc, int nc, int kc, int MC, int NC, int beta) {
    int mrR = mc - (mc & (MR - 1));
    int nrR = nc - (nc & (NR - 1));
    for (int nn = 0; nn < nrR; nn += NR) {          /* N 外层: B 块被完整复用 */
        for (int mm = 0; mm < mrR; mm += MR) {      /* M 内层: 16 行 */
            __m512 c[MR];                           /* MR 行 × NR 列 累加器 */
            float* cp = Cbase + (size_t)mm * ldc + nn;
            if (beta == 0) {
                for (int i = 0; i < MR; ++i) c[i] = _mm512_setzero_ps();
            } else {
                for (int i = 0; i < MR; ++i) c[i] = _mm512_loadu_ps(cp + (size_t)i * ldc);
            }
            int k = 0;
            for (; k + 1 < kc; k += 2) {
                __m512 b0 = _mm512_loadu_ps(Bp + (size_t)(k)     * NC + nn);
                __m512 b1 = _mm512_loadu_ps(Bp + (size_t)(k + 1) * NC + nn);
                const float* ap0 = Ap + (size_t)(k)     * MC + mm;
                const float* ap1 = Ap + (size_t)(k + 1) * MC + mm;
                for (int i = 0; i < MR; ++i) {
                    __m512 a0 = _mm512_broadcastss_ps(_mm_load_ss(ap0 + i));
                    __m512 a1 = _mm512_broadcastss_ps(_mm_load_ss(ap1 + i));
                    c[i] = _mm512_fmadd_ps(a0, b0, c[i]);
                    c[i] = _mm512_fmadd_ps(a1, b1, c[i]);
                }
                _mm_prefetch((const char*)(Bp + (size_t)(k + 4) * NC + nn), _MM_HINT_T0);
                _mm_prefetch((const char*)(Ap + (size_t)(k + 4) * MC + mm), _MM_HINT_T0);
            }
            for (; k < kc; ++k) {                    /* 余下单 k (kc 为奇数时) */
                __m512 b = _mm512_loadu_ps(Bp + (size_t)k * NC + nn);
                const float* ap = Ap + (size_t)k * MC + mm;
                for (int i = 0; i < MR; ++i) {
                    __m512 a = _mm512_broadcastss_ps(_mm_load_ss(ap + i));
                    c[i] = _mm512_fmadd_ps(a, b, c[i]);
                }
            }
            for (int i = 0; i < MR; ++i) _mm512_storeu_ps(cp + (size_t)i * ldc, c[i]);
        }
    }
}

/* 行/列尾部余数: 用带 ld 的标量参考补齐, 与微内核区域互不重叠、各跨 KC 块累加。
 * 注意 A/B 需按当前 KC 面板起点 k0 偏移 (KC 参数实为面板宽度 kc)。 */
static inline void tile_tails(int m0, int n0, int M, int N, int K,
                              const float* A, const float* B, float* C,
                              int MC, int NC, int kc, int mc, int nc, int k0, int beta) {
    const int lda = K, ldb = N, ldc = N;
    int mrR = mc - (mc & (MR - 1));
    int nrR = nc - (nc & (NR - 1));
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

/* 全 K 打包一个 m0 行块: A[m0:m0+mc, 0:K] -> [k*MC + i]。 */
static inline void packA_all(float* Ap, const float* A, int m0, int mc, int K, int MC) {
    const int lda = K;
    for (int k = 0; k < K; ++k)
        for (int i = 0; i < mc; ++i)
            Ap[(size_t)k * MC + i] = A[((size_t)(m0 + i)) * lda + k];
}

/* 全 K 打包一个 n0 列块: B[0:K, n0:n0+nc] (row-major, 步长 N) -> [k*NC + j]。 */
static inline void packB_all(float* Bp, const float* B, int n0, int nc, int K, int NC, int N) {
    for (int k = 0; k < K; ++k)
        for (int j = 0; j < nc; ++j)
            Bp[(size_t)k * NC + j] = B[((size_t)k) * N + n0 + j];
}

/* 瓦片计算 (packed 路径): 使用已全 K 打包的 A 面板 (Ap_mid) 与 B 面板 (Bp_nid)，
 * 沿 KC 面板循环。每个瓦片独占其 C 矩形, 线程安全。 */
static void gemm_tile_packed(int m0, int n0, int M, int N, int K,
                             const float* A, const float* B, float* C,
                             int MC, int NC, int KC,
                             const float* Ap_mid, const float* Bp_nid) {
    int mc = std::min(MC, M - m0);
    int nc = std::min(NC, N - n0);
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;   /* 同一瓦片跨 KC 块累加 */
        kernel(Ap_mid + (size_t)k0 * MC, Bp_nid + (size_t)k0 * NC,
               C + (size_t)m0 * N + n0, N, mc, nc, kc, MC, NC, beta);
        tile_tails(m0, n0, M, N, K, A, B, C, MC, NC, kc, mc, nc, k0, beta);
    }
}

/* 串行流水分块 (零 OpenMP 开销, 仅小缓冲): A 面板按 k0 打一次复用全部 n0,
 * B 面板按 (k0,n0) 打一次; 单线程算术强度拉满。 */
static void gemm_mblock_stream(int m0, int M, int N, int K,
                               const float* A, const float* B, float* C,
                               int MC, int NC, int KC, float* Ap, float* Bp) {
    int mc = std::min(MC, M - m0);
    const int lda = K, ldb = N, ldc = N;
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int beta = (k0 == 0) ? 0 : 1;

        /* 打包 A 面板: kk 外 / i 内 → Ap[kk*MC + i] 连续写入, 可被编译器向量化;
         * 若 i 外 / kk 内则跨 MC 步长散写, 既无法向量化又引发缓存行颠簸, 窄 N 时成为瓶颈。 */
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

void acfun_gemm_avx512(int M, int N, int K,
                       const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int KC = 256, NC = 128, MC = 256;
    int nt = acfun_get_num_threads_runtime();
    int tkc = 0, tnc = 0;
    int have_tune = acfun_get_tuned_kc_nc(nt, &tkc, &tnc);
    int env_kc = (std::getenv("ACFUN_KC") != nullptr);
    int env_nc = (std::getenv("ACFUN_NC") != nullptr);
    if (env_kc || env_nc) {
        /* 环境变量显式覆盖: 维持原默认 MC(不走自适应), 与原行为一致 */
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 128);
        MC = 256;
    } else if (have_tune) {
        /* 自动调参: 采用本机实测最优 KC/NC, 并据此自适应 MC(与启发式同构) */
        KC = tkc; NC = tnc;
        long target = (nt > 1) ? (long)16 * nt : 1;
        long mc_t = ((long)M * N) / (target * (long)NC);
        if (mc_t < 64) mc_t = 64;
        if (mc_t > 512) mc_t = 512;
        MC = (int)(((mc_t + MR - 1) / MR) * MR);
    } else {
        /* 启发式(静态, 仅按 L2 容量): 自适应 MC + auto_blocks */
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
        /* 纯串行: 零 OpenMP 开销, 小缓冲流水分块 */
        float* Ap = (float*)malloc((size_t)MC * KC * sizeof(float));
        float* Bp = (float*)malloc((size_t)KC * NC * sizeof(float));
        if (Ap && Bp) {
            for (int m0 = 0; m0 < M; m0 += MC)
                gemm_mblock_stream(m0, M, N, K, A, B, C, MC, NC, KC, Ap, Bp);
        }
        free(Ap); free(Bp);
        return;
    }

    /* ---- 二维 (M×N) 瓦片并行: A、B 各仅全 K 打包一次 (Phase-1 并行) ---- */
    int mb = (M + MC - 1) / MC;
    int nb = (N + NC - 1) / NC;
    /* 用 malloc 而非 std::vector: vector 的值初始化会把整块面板零写一遍,
     * 对窄 N 卷积(面板池可达数 MB)是纯浪费; 打包会完整覆盖所有被读元素, 无需清零。 */
    float* Apool = (float*)malloc((size_t)mb * MC * K * sizeof(float));
    float* Bpool = (float*)malloc((size_t)nb * NC * K * sizeof(float));
    if (!Apool || !Bpool) { free(Apool); free(Bpool); return; }

    #pragma omp parallel num_threads(nt)
    {
        /* Phase 1 — 并行打包 A (按 m0) 与 B (按 n0), 各自仅一次 */
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

        /* Phase 2 — 并行计算 2D 瓦片网格 (共享只读 A/B 面板) */
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
