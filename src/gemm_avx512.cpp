/* Acfun GEMM — AVX-512F + FMA 后端 (GotoBLAS 风格三级分块)
 *
 * 关键优化 (对标 MNN / 一线 BLAS 的工程手段):
 *   1) 数据打包: A 面板按 k 连续重排 ([kc*MC + i]); B 面板同样按 k 连续重排到
 *      [kc*NC + j] 的连续缓冲区, 使微内核沿 k 的 B 访问完全顺序化, 彻底消除
 *      对 B 的 stride-N 跨步访问, B 工作集常驻 L2。
 *   2) 三级分块: MC(命中 L3) × KC(命中 L2) × NC(命中 L1), 微内核工作集 (MR+NR)*KC
 *      仅 ~(MR+NR)*KC*4 字节, 稳稳落在 L1, 算术强度高。
 *   3) 微内核循环序 N 外层 / M 内层: 每个 16 列块载入一次 B 面板行, 被该列块下
 *      全部 16 行块复用, B 访存大幅下降; 累加器跨整个 KC 累加, β=1 跨 KC 块累加
 *      保证大矩阵正确性。
 *   4) β=1 跨 KC 累加: C 的同一 (M,N) 瓦片跨多个 KC 块累加, k0==0 块以 β=0 覆盖、
 *      其余以 β=1 累加, 单精度无误差累积。
 *   5) 二维 (M×N) 瓦片并行 (对标 MNN 的多维并行): C 被划分为 mb×nb 个矩形瓦片,
 *      以 mb*nb 个任务铺满全部核。关键 —— A 面板每个 (m0) 宏块**仅打包一次**
 *      (并行 phase-1 完成, 全 K 打包进共享只读缓冲), 随后 phase-2 在 2D 网格上并行
 *      计算, 所有 n0 列块复用同一份只读 A 面板; 既获得 N 维并行 (任务数较仅沿 M
 *      并行提升 N/NC 倍, 多核利用率大幅上升), 又彻底消除冗余 A 打包, 大 K 矩阵依旧高效。
 *   6) 自适应 MC: 以 ~16 任务/线程为目标选取瓦片大小 —— 大矩阵用大瓦片(降调度开销),
 *      小矩阵用小瓦片(填满多核), 使 2D 任务数随矩阵与核数自适应 (对标 MNN 调参)。
 *   7) 打包缓冲线程私有、跨任务复用: 每个线程持有一对 (A面板,B面板) 缓冲, 在全部
 *      任务间复用, 消除并行区内的重复 malloc/free 开销 (小矩阵尤为关键)。
 *   8) 串行快路径: 仅 1 线程时 A 面板在 (m0) 处仅打包一次、复用全部 n0, 且零 OpenMP
 *      开销, 单线程算术强度拉满。
 *
 * 微内核只处理完整 16×16 块; 行/列尾部余数交给标量参考 (β=0/β=1 两套), 保证正确。
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

static inline void kernel_16x16(const float* Ap, const float* Bp,
                                float* C, int ldc,
                                int mc, int nc, int kc,
                                int MC, int NC, int beta) {
    int mrR = mc - (mc & 15);
    int nrR = nc - (nc & 15);
    for (int nn = 0; nn < nrR; nn += 16) {           /* N 外层: B 行被复用 */
        for (int mm = 0; mm < mrR; mm += 16) {       /* M 内层 */
            __m512 c[16];
            if (beta == 0) {
                for (int i = 0; i < 16; ++i) c[i] = _mm512_setzero_ps();
            } else {
                for (int i = 0; i < 16; ++i)
                    c[i] = _mm512_loadu_ps(C + ((size_t)(mm + i)) * ldc + nn);
            }
            for (int k = 0; k < kc; ++k) {
                __m512 b = _mm512_loadu_ps(Bp + ((size_t)k) * NC + nn);
                for (int i = 0; i < 16; ++i) {
                    float a = Ap[((size_t)k) * MC + mm + i];   /* A 连续 16 值 */
                    c[i] = _mm512_fmadd_ps(_mm512_set1_ps(a), b, c[i]);
                }
            }
            for (int i = 0; i < 16; ++i)
                _mm512_storeu_ps(C + ((size_t)(mm + i)) * ldc + nn, c[i]);
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

        int mrR = mc - (mc & 15), nrR = nc - (nc & 15);
        kernel_16x16(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc,
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

/* 串行 m0 宏块: A 面板 (m0,k0) 仅打一次, 复用于该 m0 下所有 n0 列块 (零冗余)。 */
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

            int mrR = mc - (mc & 15), nrR = nc - (nc & 15);
            kernel_16x16(Ap, Bp, C + (size_t)m0 * ldc + n0, ldc,
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
}

void acfun_gemm_avx512(int M, int N, int K,
                       const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    const int MR = 16, NR = 16;
    int KC = 256, NC = 128;
    int MC = 256;
    int nt = acfun_get_num_threads_runtime();
    if (std::getenv("ACFUN_KC") || std::getenv("ACFUN_NC")) {
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 128);
    } else {
        /* 自适应 MC: 以 ~16 任务/线程为目标选取瓦片大小 —— 大矩阵用大瓦片(降调度开销),
         * 小矩阵用小瓦片(填满多核), 使 2D 任务数随矩阵与核数自适应。 */
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
        /* 纯串行: 零 OpenMP 开销, A 仍只打一次 */
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
     * 否则走 2D 瓦片并行 (缓存可容, 2D 并行收益显著)。二者均保持正确性。 */
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    bool membound = (l3 > 0) && ((long)M * K * 4L > l3);

    if (membound) {
        /* m0 宏块并行: 每个 m0 内 A 仅打一次、复用全部 n0; 无 A 池, 不增加内存写流量。
         * 小 MC 以获取充足 m0 任务数铺满多核 (此路径无额外打包开销)。 */
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
    /* A 面板池: 每个 m0 宏块全 K 打包一次, 跨全部 n0 列块只读复用 */
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
