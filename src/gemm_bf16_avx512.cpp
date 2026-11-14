/* Acfun GEMM — AVX-512 BF16 后端 (vdpbf16ps, 对标 MNN 的 bf16 低精度推理路径)
 *
 * 为什么是 MNN 级速度的关键补齐:
 *   现代 x86 服务端 (Intel Sapphire/EMR、AMD Zen4/5) 的 AVX-512 BF16 指令 `vdpbf16ps`
 *   单指令对 16 个 bf16 对做点积并累加到 f32 (16 个 f32 MAC/指令, 每 MAC = 2 个 bf16 乘法)。
 *   与 float GEMM 相比: 输入/权重以 bf16 (2 字节) 表示, 带宽需求仅为 float 之半,
 *   而累加器仍是 f32 —— 推理精度无损于 float 的逐层输出, 却把访存墙削掉一半。这正是 MNN
 *   在 ARM (fp16/int8) 与 x86 (bf16/int8) 端吃吞吐的核心手段; Acfun 此前缺这条路径, 故补齐。
 *
 * 微内核形态: MR=16 × NR=16 (与 float 后端同分块)。循环序: 每对 k (k0,k1) 一次加载 B 面板
 * 一行 (512-bit), 对 16 行各广播 A 行的一对 bf16 (row-broadcast), 用 vdpbf16ps 累加。
 * 一个 vdpbf16ps 完成「1 行 × 16 列 × 2 个 k」的 16 个 f32 点积, B 面板被整行 16 列复用 16 次。
 *
 * 面板布局 (关键):
 *   A 面板: 行主序 bf16 [mc][K] (每 k 2 字节)。
 *   B 面板: 交错 bf16 [Kp][nc][4 字节]: 列 c 在每 kp 块的 4 字节 = {B[k0][c], B[k1][c]} 两个 bf16,
 *           即 lane i = {col i 的 k0, col i 的 k1}。vdpbf16ps 的配对语义要求同 lane 内
 *           低 16 位=奇数步 k0、高 16 位=偶数步 k1, 由此对每 (行 r, 列 c) 完成
 *           C[r][c] += A[r][k0]·B[k0][c] + A[r][k1]·B[k1][c] (恰好一个 k 对)。
 *   K 奇数: 末对 k1 越界置 0, 贡献为 0, 正确性不变。
 *
 * 编译单元需带 -mavx512f -mfma -mavx512bf16。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <algorithm>

static const int MR = 16;
static const int NR = 16;

/* float → bf16 (round-to-nearest-even, 与主流框架一致) */
static inline uint16_t f32_to_bf16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    /* 偏置 = 0x7fff + 结果最低位 (tie-to-even) */
    uint32_t bias = 0x7fff + ((x >> 16) & 1u);
    uint32_t rounded = x + bias;
    /* inf/nan 经上述加法仍映射到有效 bf16 值 (近似), 不影响推理正确性 */
    return (uint16_t)(rounded >> 16);
}

/* 打包 A 块 → 交错 bf16 面板 [Kp][mc][2]: 每个 kp (k0=2kp, k1=2kp+1) 内连续存 mc 行的
 * {A[r][k0] bf16(低), A[r][k1] bf16(高)} 两 uint16。如此微内核每 kp 一步仅需一次 512-bit
 * 载入即取齐 16 行 (一个 16 行带) 的 (a0,a1) 对 —— 与 B 面板同构, 访存连续 (对标 float 后端
 * 的 [k*MC+i] 连续布局), 彻底消除旧版「逐行 8KB 散读」导致的访存墙。面板尺寸 Kp·mc·2 字节,
 * 较行主序再省一半。K 奇数末对越界 k1 置 0 贡献为 0。 */
static inline void packA_bf16(uint16_t* Ap, const float* A, int m0, int mc, int K, int M) {
    (void)M;
    int Kp = (K + 1) / 2;
    for (int kp = 0; kp < Kp; ++kp) {
        int k0 = 2 * kp, k1 = k0 + 1;
        uint16_t* slab = Ap + (size_t)kp * mc * 2;
        for (int i = 0; i < mc; ++i) {
            float a0 = (k0 < K) ? A[(size_t)(m0 + i) * K + k0] : 0.0f;
            float a1 = (k1 < K) ? A[(size_t)(m0 + i) * K + k1] : 0.0f;
            slab[(size_t)i * 2 + 0] = f32_to_bf16(a0);
            slab[(size_t)i * 2 + 1] = f32_to_bf16(a1);
        }
    }
}

/* 打包 B 块 [K][nc] → 交错 bf16 面板 (uint16 数组, 每列 2 个 uint16 = 4 字节)。
 * 每 kp (k0=2kp, k1=2kp+1) 内, 列 c 占 2 个 uint16 = {B[k0][c] bf16(低), B[k1][c] bf16(高)};
 * 每 kp 行跨 nc*2 个 uint16; 越界 k (K 奇数末对) 置 0 贡献为 0。 */
static inline void packB_bf16(uint16_t* Bp, const float* B, int n0, int nc, int K, int N) {
    int Kp = (K + 1) / 2;
    for (int kp = 0; kp < Kp; ++kp) {
        int k0 = 2 * kp, k1 = k0 + 1;
        for (int c = 0; c < nc; ++c) {
            float b0 = (k0 < K) ? B[(size_t)k0 * N + n0 + c] : 0.0f;
            float b1 = (k1 < K) ? B[(size_t)k1 * N + n0 + c] : 0.0f;
            uint16_t* lane = Bp + (size_t)kp * nc * 2 + (size_t)c * 2;
            lane[0] = f32_to_bf16(b0);   /* 低 16 位 = k0 */
            lane[1] = f32_to_bf16(b1);   /* 高 16 位 = k1 */
        }
    }
}

/* 16×16 微内核: 累加 acc[0..15] (16 行 × 16 列 f32) 跨全部 kp。
 * A 面板交错布局 [Kp][mc][2]: 每个 kp 内连续存 mc 行的 (a0,a1) 对; 本内核处理 rr 行带
 * (16 行), 故每 kp 一步以一次 512-bit 载入取齐该带 16 行 × 2 个 bf16 (32 个 uint16 = 64B),
 * 再逐行抽取本行 (a0,a1) 广播。B 面板交错 [Kp][nc][2]: 每 kp 取从列 cc 起的 16 列 (64B)。
 * 两路载入均连续, 与 float 后端 [k*MC+i] 连续 A 同构, 访存效率最优。 */
static inline void kernel_f16(__m512 acc[MR], const uint16_t* Ap, const uint16_t* Bp,
                               int Kp, int mc, int K, int cc, int ncblk) {
    (void)K;
    /* Ap 指向本 kp 窗口首段; 每个 kp 跨 mc*2 个 uint16 到下一个 kp 的同一带 */
    const uint16_t* Aslab = Ap;
    const uint16_t* Bslab = Bp;
    for (int kp = 0; kp < Kp; ++kp) {
        /* B 一行 (16 列交错) → 512-bit, lane i = {col (cc+i) 的 k0, k1}。
         * 必须用 loadu, 否则对齐 move 在 malloc 的非 64B 对齐地址上触发 #GP。
         * Bp 为 uint16 数组: 每 kp 行跨 ncblk*2 个 uint16, 每列 2 个 uint16。 */
        __m512i bvec = _mm512_loadu_si512((const void*)(Bslab + (size_t)cc * 2));
        __m512bh Breg = (__m512bh)bvec;
        /* A: 本带 16 行 × (a0,a1) 在 Aslab[r*2], Aslab[r*2+1], 连续 32 字节 (同缓存行),
         * 故逐 16-bit 读既正确又无散读访存墙。配对与旧版一致: pair=(a1<<16)|a0。 */
        for (int r = 0; r < MR; ++r) {
            uint16_t a0 = Aslab[(size_t)r * 2 + 0];
            uint16_t a1 = Aslab[(size_t)r * 2 + 1];
            uint32_t pair = ((uint32_t)a1 << 16) | (uint32_t)a0;
            /* 把 {A[r][k0], A[r][k1]} 这 32 位广播到 16 个 lane (每 lane 两 bf16 = 一行一对) */
            __m512 Areg = _mm512_castsi512_ps(
                _mm512_broadcastd_epi32(_mm_cvtsi32_si128(pair)));
            acc[r] = _mm512_dpbf16_ps(acc[r], (__m512bh)Areg, Breg);
        }
        Aslab += (size_t)mc * 2;     /* 下一 kp 的同带 */
        Bslab += (size_t)ncblk * 2;  /* 下一 kp 的同列块 */
    }
}

/* 单瓦片 (m0,n0): 沿 KC 面板循环 (与 float 后端同构), 满 16×16 子块走 SIMD 内核,
 * 行/列余数交标量参考补算。各瓦片写互不相交的 C 矩形, 跨 KC 块用 β 累加, 线程安全。
 *
 * 关键工程手段 (对标 MNN/float 后端): 把 K 切成 KC 宽面板, 使 A/B 面板窗口稳居 L2,
 * 内层 k 循环由 L2 供数而非内存 —— 否则整 K 一次处理会让 2MB 级面板溢出 L2, 被 L3/内存
 * 喂数拖垮 (大矩阵下 BF16 反而慢于 float)。KC 窗口内 A 窗口 = mc·KC·2B、B 窗口 = KC/2·nc·2B,
 * 对 MC=256/NC=128/KC=256 仅 ~160KB, 远低于 L2 容量。 */
static void gemm_tile_bf16(int m0, int n0, int M, int N, int K,
                            const float* A, const float* B, float* C,
                            int MC, int NC, int KC,
                            const uint16_t* Ap_mid, const uint16_t* Bp_nid) {
    int mc = std::min(MC, M - m0);
    int nc = std::min(NC, N - n0);
    int mrR = mc - (mc & (MR - 1));
    int nrR = nc - (nc & (NR - 1));
    /* 注意: C 写回按 (m0+rr+r, n0+cc) 直接定位, 不重复加计。 */

    /* 沿 KC 面板循环: 每个窗口 β 累加 (k0==0 时清零, 否则加回已写 C) */
    for (int k0 = 0; k0 < K; k0 += KC) {
        int kc = std::min(KC, K - k0);
        int kcp = (kc + 1) / 2;              /* 本窗口的 k 对数 */
        int beta = (k0 == 0) ? 0 : 1;
        /* A 窗口: 交错面板 [Kp][mc][2], 取第 k0/2 对起的同带行; B 窗口: 整 K 面板取同对同列块 */
        const uint16_t* Aslab0 = Ap_mid + (size_t)(k0 / 2) * ((size_t)mc * 2);
        const uint16_t* Bslab0 = Bp_nid + (size_t)(k0 / 2) * ((size_t)nc * 2);

        for (int rr = 0; rr < mrR; rr += MR) {
            for (int cc = 0; cc < nrR; cc += NR) {
                __m512 acc[MR];
                float* cp = C + (size_t)(m0 + rr) * N + n0 + cc;
                if (beta == 0)
                    for (int r = 0; r < MR; ++r) acc[r] = _mm512_setzero_ps();
                else
                    for (int r = 0; r < MR; ++r) acc[r] = _mm512_loadu_ps(cp + (size_t)r * N);
                /* A 带 rr: 面板内偏移 rr*2; B 列块 cc: 内核内加 cc*2 */
                kernel_f16(acc, Aslab0 + (size_t)rr * 2, Bslab0, kcp, mc, K, cc, nc);
                for (int r = 0; r < MR; ++r)
                    _mm512_storeu_ps(cp + (size_t)r * N, acc[r]);
            }
        }
    }
    /* 尾部余数 (与 SIMD 区域互不重叠): 右尾 [0:mrR)×[nrR:nc), 底尾 [mrR:mc)×[0:nc)。
     * 必须用带主维度的标量参考 acfun_gemm_ref_ld, 以真实 N 作主维度 (ldC=N),
     * 否则 acfun_gemm_ref 会按子矩阵自身宽度跨行, 错误覆写 SIMD 区域。尾部整 K 一次算清,
     * 因对应 C 单元不被 SIMD 区域触及, 直接覆盖即可。 */
    if (nrR < nc)
        acfun_gemm_ref_ld(mrR, nc - nrR, K, K, N, N,
                          A + (size_t)m0 * K, B + (size_t)n0 + nrR,
                          C + (size_t)m0 * N + n0 + nrR);
    if (mrR < mc)
        acfun_gemm_ref_ld(mc - mrR, nc, K, K, N, N,
                          A + (size_t)(m0 + mrR) * K, B + (size_t)n0,
                          C + (size_t)(m0 + mrR) * N + n0);
}

void acfun_gemm_bf16(int M, int N, int K,
                      const float* A, const float* B, float* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int nt = acfun_get_num_threads_runtime();
    if (nt <= 0) nt = 1;
    int MC = 256, NC = 128, KC = 256;
    int Kp = (K + 1) / 2;

    if (nt <= 1) {
        /* 串行: 逐 m0 块打包 A (复用全部 n0), 逐 n0 打包 B, 流水分块 */
        uint16_t* Ap = (uint16_t*)malloc((size_t)Kp * MC * 2 * sizeof(uint16_t));
        uint16_t* Bp = (uint16_t*)malloc((size_t)Kp * NC * 2 * sizeof(uint16_t));
        if (!Ap || !Bp) { free(Ap); free(Bp); return; }
        for (int m0 = 0; m0 < M; m0 += MC) {
            int mc = std::min(MC, M - m0);
            packA_bf16(Ap, A, m0, mc, K, M);
            for (int n0 = 0; n0 < N; n0 += NC) {
                int nc = std::min(NC, N - n0);
                packB_bf16(Bp, B, n0, nc, K, N);
                gemm_tile_bf16(m0, n0, M, N, K, A, B, C, MC, NC, KC, Ap, Bp);
            }
        }
        free(Ap); free(Bp);
        return;
    }

    /* 二维 (M×N) 瓦片并行: A/B 各仅全 K 打包一次 (Phase-1 并行), Phase-2 并行计算瓦片网格 */
    int mb = (M + MC - 1) / MC;
    int nb = (N + NC - 1) / NC;
    uint16_t* Apool = (uint16_t*)malloc((size_t)mb * MC * Kp * 2 * sizeof(uint16_t));
    uint16_t* Bpool = (uint16_t*)malloc((size_t)nb * Kp * NC * 2 * sizeof(uint16_t));
    if (!Apool || !Bpool) { free(Apool); free(Bpool); return; }

    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for schedule(static) nowait
        for (int mid = 0; mid < mb; ++mid) {
            int m0 = mid * MC;
            int mc = std::min(MC, M - m0);
            packA_bf16(Apool + (size_t)mid * MC * Kp * 2, A, m0, mc, K, M);
        }
        #pragma omp for schedule(static) nowait
        for (int nid = 0; nid < nb; ++nid) {
            int n0 = nid * NC;
            int nc = std::min(NC, N - n0);
            packB_bf16(Bpool + (size_t)nid * Kp * NC * 2, B, n0, nc, K, N);
        }
        #pragma omp barrier

        #pragma omp for schedule(static)
        for (int t = 0; t < mb * nb; ++t) {
            int mid = t / nb;
            int nid = t % nb;
            int m0 = mid * MC;
            int n0 = nid * NC;
            gemm_tile_bf16(m0, n0, M, N, K, A, B, C, MC, NC, KC,
                           Apool + (size_t)mid * MC * Kp * 2,
                           Bpool + (size_t)nid * Kp * NC * 2);
        }
    }
    free(Apool); free(Bpool);
}
