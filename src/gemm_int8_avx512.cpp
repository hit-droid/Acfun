/* Acfun INT8 GEMM — AVX-512F + VNNI 后端 (对标 MNN int8 推理)
 *
 * `vpdpbusd` 语义: 通道 ℓ (0..15) 累加 = Σ_{t=0..3} a[4ℓ+t] (unsigned) · b[4ℓ+t] (signed)。
 * 一次指令算完「4 行 × 4 列 × 连续 4 个 K」的点积并累加到 16 个 int32 累加器 (通道 ℓ
 * 对应 (行=ℓ/4, 列=ℓ%4))。故面板须用「扩展 4×4 布局」打包为 64 字节块 (每块覆盖 4 行/4 列):
 *   A: 块内字节 p (p=0..63) -> 行 = base + p/16, k = p%4      (行随 ℓ/4 变化)
 *   B: 块内字节 p -> 列 = base + (p/4)%4, k = p%4              (列随 ℓ%4 变化)
 * 这样 64 字节 zmm 一次 dpbusd 即更新 4×4 块全部 16 个累加器 (64 个 int8 MAC/指令),
 * 算力密度较 float 提升 16×、带宽需求降为 1/4, 正是 MNN 在 x86 端拿 int8 吞吐的标准手法。
 *
 * 三级分块 + 2D 瓦片并行 (A/B 各全 K 打包一次) 沿用 float GEMM 的工程结构。关键正确性
 * 约束 (与 float GEMM 一致): 打包步长与微内核索引步长使用「固定块数」MCblk/NCblk (由
 * MC/NC 推出, 对所有瓦片恒定), 仅 mc/nc 用于约束有效行/列与循环边界 —— 绝不可让打包步长
 * 与内核步长分别取「满块」与「实际块」, 否则尾瓦片 (mc<MC 或 nc<NC) 会偏移到错误位置。
 * 输出为 int32 累加和, 反量化 (× s_in × 每通道 scale) 由卷积层完成。编译单元需带
 * -mavx512f -mfma -mavx512vnni。 */
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

static void auto_blocks(int MC, int* KC, int* NC) {
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long cap = (l2 > 0) ? (l2 / 2) : (1L << 19);
    int kc = 256, nc = 256;
    while ((long)kc * nc + (long)MC * kc > cap && kc > 64) kc /= 2;
    while ((long)kc * nc + (long)MC * kc > cap && nc > 64) nc /= 2;
    *KC = kc; *NC = nc;
}

static const int MR = 4;    /* 微内核行块 = dpbusd 通道高 2 位 (ℓ/4) */
static const int NR = 4;    /* 微内核列块 = dpbusd 通道低 2 位 (ℓ%4) */
static const int KG = 4;    /* 每次 dpbusd 覆盖的连续 K 数 */
static const int CHUNK = 64;/* 每块 4 行/4 列 × 16 字节 */
/* K 展开因子 UU: 覆盖连续 UU 个 k 组, 与 GB×UC 个独立累加器共同暴露指令级并行。
 * 列展开 UC: 一次处理 UC 个 4 列块; 行组 GB: 一次处理 GB 个 4 行块。二者配合使 B
 * 列块载入寄存器后跨 GB 个行块复用 —— 把「每 dpbusd 的负载比」从 2 (A+B 都从内存) 降到
 * ~0.5 (B 摊销到 GB×MCblk 行, 仅 A 随行重载)。dpbusd 带内存操作数实测峰值仅 ~80-130
 * GOPS (受 2 个 load 端口限制), 必须让 B 驻留寄存器并多累加器并行才逼近该上限。 */
static const int UU = 2;   /* K 展开 */
static const int UC = 4;   /* 列展开: 同时算 UC 个 4 列块 */
static const int GB = 4;   /* 行组: 同时算 GB 个 4 行块 (B 随之跨行复用) */

/* 微内核: 处理 (m0,m0+mc) × (n0,n0+nc) 块, K 按 4 分组且 Kg 已对齐为 UU 的倍数。
 * 块布局 (blk*Kg + kg): 同一 blk 的连续 kg 相邻, 利于 K 展开连续流式读取。
 * mc/nc 仅用于约束有效行/列与循环边界。返回写入 Cbase[ib+i, jb+j] (ldc 行步长)。 */
static inline void kernel_int8(const uint8_t* Ap, const int8_t* Bp, int32_t* Cbase, int ldc,
                              int mc, int nc, int Kg, int MCblk, int NCblk) {
    (void)MCblk; (void)NCblk;
    for (int jb = 0; jb < nc; jb += NR * UC) {
        int jbR = std::min(NR * UC, nc - jb);   /* 本组实际列数 */
        int cR = (jbR + NR - 1) / NR;            /* 有效 4 列块数 (向上取整, 含尾部部分块) */
        int cbase = jb / NR;
        for (int ib = 0; ib < mc; ib += MR * GB) {
            int rbG = ib / MR;                    /* 起始行块索引 */
            int iRg = std::min(MR * GB, mc - ib); /* 本组实际行数 */
            int gR = (iRg + MR - 1) / MR;         /* 有效行块数 (向上取整, 含尾部部分块) */
            __m512i acc[GB][UC];
            for (int g = 0; g < GB; ++g)
                for (int c = 0; c < UC; ++c) acc[g][c] = _mm512_setzero_si512();
            for (int kg = 0; kg < Kg; kg += UU) {
                /* 载入本 k 步全部列块的 B (驻留寄存器, 跨后续 GB 个行块复用) */
                __m512i b[UC][UU];
                for (int c = 0; c < cR; ++c)
                    for (int k = 0; k < UU; ++k) {
                        b[c][k] = _mm512_loadu_si512((const __m512i*)(Bp + (size_t)((cbase + c) * Kg + kg + k) * CHUNK));
                        _mm_prefetch((const char*)(Bp + (size_t)(((cbase + c + UC) * Kg + kg + k + UU) * CHUNK)), _MM_HINT_T0);
                    }
                for (int g = 0; g < gR; ++g) {
                    int rb = rbG + g;            /* 当前行块索引 */
                    __m512i a[UU];
                    for (int k = 0; k < UU; ++k)
                        a[k] = _mm512_loadu_si512((const __m512i*)(Ap + (size_t)(rb * Kg + kg + k) * CHUNK));
                    for (int c = 0; c < cR; ++c)
                        for (int k = 0; k < UU; ++k)
                            acc[g][c] = _mm512_dpbusd_epi32(acc[g][c], a[k], b[c][k]);
                }
            }
            /* 写回: 各 (行块, 列块) 的 4×4 块按有效行/列约束 scatter 到 C */
            for (int g = 0; g < gR; ++g) {
                int rb = rbG + g;
                for (int c = 0; c < cR; ++c) {
                    int jcol = jb + c * NR;
                    int jR = std::min(NR, nc - jcol);
                    int32_t tmp[16];
                    _mm512_storeu_si512((__m512i*)tmp, acc[g][c]);
                    for (int L = 0; L < 16; ++L) {
                        int i = L >> 2, j = L & 3;
                        if (ib + g * MR + i < mc && j < jR)
                            Cbase[((size_t)(ib + g * MR + i)) * ldc + (jcol + j)] = tmp[L];
                    }
                }
            }
        }
    }
}

/* 把 4 个 k 字节 (b0..b3) 展开为 16 字节: [b0 b1 b2 b3 b0 b1 b2 b3 b0 b1 b2 b3 b0 b1 b2 b3]。
 * 微内核读取字节 16r+4c+t 应落到 A 的 v(r,t)=b_t (与列 c 无关); 本布局使 (4c+t) 处的字节恰为 b_t,
 * 与旧打包逐字节写等价, 但一次 128-bit 存储即覆盖一个 (rb,kg) 块的 16 行字节, 较逐字节写 + memset 提速一个量级。 */
static inline __m128i broadcast4x4(uint32_t k4) {
    uint8_t b0 = (uint8_t)k4, b1 = (uint8_t)(k4 >> 8), b2 = (uint8_t)(k4 >> 16), b3 = (uint8_t)(k4 >> 24);
    return _mm_set_epi8(b3,b2,b1,b0, b3,b2,b1,b0, b3,b2,b1,b0, b3,b2,b1,b0);
}

/* 打包一个 m0 行块 (扩展 4×4): 块内 p -> A[m0 + rb*MR + p/16, kg*KG + p%4]。
 * 块布局 (rb*Kg + kg): 同一 rb 的连续 kg 块在内存中相邻, 供 K 展开连续流式读取。
 * 仅 row<mc 视为有效, 其余零填充。
 * 提速: 整块有效 (全部 4 行与 4 个 k 落在校验内) 时, 对每一行取 4 个连续 k 字节构造成 16 字节
 * 广播扇区, 4 次 128-bit 存储即覆盖整块 (旧实现为 memset(64) + 64 次逐字节写); 仅尾部块回落零感知标量。 */
static inline void packA_int8(uint8_t* Ap, const uint8_t* A, int m0, int mc, int K, int Kup, int MCblk) {
    int Kg = Kup / KG;
    for (int rb = 0; rb < MCblk; ++rb) {
        uint8_t* base = Ap + (size_t)rb * Kg * CHUNK;
        int row0 = m0 + rb * MR;
        bool rows_ok = (row0 + MR <= mc);
        for (int kg = 0; kg < Kg; ++kg) {
            uint8_t* cp = base + (size_t)kg * CHUNK;
            bool k_ok = ((kg + 1) * KG <= K);
            if (rows_ok && k_ok) {
                for (int r = 0; r < MR; ++r) {
                    uint32_t k4 = *(const uint32_t*)&A[((size_t)(row0 + r)) * K + (size_t)kg * KG];
                    _mm_storeu_si128((__m128i*)(cp + 16 * r), broadcast4x4(k4));
                }
            } else {
                for (int p = 0; p < CHUNK; ++p) {
                    int row = rb * MR + (p / 16);
                    int k = kg * KG + (p % 4);
                    int v = (row < mc && k < K) ? (int)A[((size_t)(m0 + row)) * K + k] : 0;
                    cp[p] = (uint8_t)v;
                }
            }
        }
    }
}
/* 打包一个 n0 列块 (扩展 4×4): 块内 p -> B[kg*KG + p%4, n0 + cb*NR + (p/4)%4]。
 * 块布局 (cb*Kg + kg): 连续 kg 相邻。仅 col<nc 视为有效, 其余零填充。
 * 提速: 整块有效时, 对每一输出列 c 跨 4 个 k 做 (stride-N) 聚集得 32 位字, 再复制到 4 个行复制位
 * (16 次 uint32 存储覆盖整块); 较 memset(64) + 64 次逐字节写大幅减少内存事务。尾部块回落标量。 */
static inline void packB_int8(int8_t* Bp, const int8_t* B, int n0, int nc, int K, int N, int Kup, int NCblk) {
    int Kg = Kup / KG;
    for (int cb = 0; cb < NCblk; ++cb) {
        int8_t* base = Bp + (size_t)cb * Kg * CHUNK;
        int col0 = n0 + cb * NR;
        bool cols_ok = (col0 + NR <= nc);
        for (int kg = 0; kg < Kg; ++kg) {
            int8_t* cp = base + (size_t)kg * CHUNK;
            bool k_ok = ((kg + 1) * KG <= K);
            if (cols_ok && k_ok) {
                for (int c = 0; c < NR; ++c) {
                    int col = col0 + c;
                    const int8_t* bp = B + (size_t)kg * KG * N + col;
                    uint32_t b0 = (uint32_t)(uint8_t)bp[0 * N];
                    uint32_t b1 = (uint32_t)(uint8_t)bp[1 * N];
                    uint32_t b2 = (uint32_t)(uint8_t)bp[2 * N];
                    uint32_t b3 = (uint32_t)(uint8_t)bp[3 * N];
                    uint32_t w = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
                    for (int r = 0; r < MR; ++r)
                        *(uint32_t*)(cp + 16 * r + 4 * c) = w;
                }
            } else {
                for (int p = 0; p < CHUNK; ++p) {
                    int col = cb * NR + ((p / 4) % 4);
                    int k = kg * KG + (p % 4);
                    int v = (col < nc && k < K) ? (int)B[((size_t)k) * N + n0 + col] : 0;
                    cp[p] = (int8_t)v;
                }
            }
        }
    }
}

static inline void run_block_int8(int m0, int n0, int M, int N, int K,
                                  const uint8_t* A, const int8_t* B, int32_t* C,
                                  int MC, int NC, int Kup, int MCblk, int NCblk,
                                  const uint8_t* Ap_mid, const int8_t* Bp_nid) {
    int mc = std::min(MC, M - m0);
    int nc = std::min(NC, N - n0);
    kernel_int8(Ap_mid, Bp_nid, C + (size_t)m0 * N + n0, N, mc, nc, Kup / KG, MCblk, NCblk);
}

/* 串行流水分块 (零 OpenMP 开销): 固定步长 MCblk/NCblk 同时用于打包与内核。 */
static void gemm_mblock_int8_stream(int m0, int M, int N, int K,
                                    const uint8_t* A, const int8_t* B, int32_t* C,
                                    int MC, int NC, int Kup, int MCblk, int NCblk,
                                    uint8_t* Ap, int8_t* Bp) {
    int mc = std::min(MC, M - m0);
    for (int n0 = 0; n0 < N; n0 += NC) {
        int nc = std::min(NC, N - n0);
        /* 固定步长 NCblk 同时用于打包与下方内核索引 (绝不可用实际 nc_blk) */
        packB_int8(Bp, B, n0, nc, K, N, Kup, NCblk);
        /* A 面板已在外部按固定步长 MCblk 打包, 此处内核同样用 MCblk 索引 */
        kernel_int8(Ap, Bp, C + (size_t)m0 * N + n0, N, mc, nc, Kup / KG, MCblk, NCblk);
    }
}

void acfun_gemm_int8_avx512(int M, int N, int K,
                            const uint8_t* A, const int8_t* B, int32_t* C) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    int KC = 256, NC = 256, MC = 256;
    int nt = acfun_get_num_threads_runtime();
    if (std::getenv("ACFUN_KC") || std::getenv("ACFUN_NC")) {
        KC = g_dim("ACFUN_KC", 256);
        NC = g_dim("ACFUN_NC", 256);
    } else {
        int tkc = 0, tnc = 0;
        if (acfun_get_tuned_kc_nc_int8(nt, &tkc, &tnc)) {
            /* 自动调参: 采用本机实测最优 KC/NC (与 float 同构的 L2 面板拟合 + 无回归门控) */
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
    }
    MC = (MC / MR) * MR; if (MC < MR) MC = MR;
    NC = (NC / NR) * NR; if (NC < NR) NC = NR;
    /* K 上对齐到 KG*UU, 使 Kg=Kup/KG 恒为 UU 的整数倍 (微内核按 UU 展开) */
    int Kup = ((K + KG * UU - 1) / (KG * UU)) * (KG * UU);
    int MCblk = (MC + MR - 1) / MR;   /* 固定 A 步长块数 (所有瓦片一致) */
    int NCblk = (NC + NR - 1) / NR;   /* 固定 B 步长块数 (所有瓦片一致) */

    if (nt <= 1) {
        uint8_t* Ap = (uint8_t*)malloc((size_t)Kup / KG * MCblk * CHUNK + 16);
        int8_t*  Bp = (int8_t*) malloc((size_t)Kup / KG * NCblk * CHUNK + 16);
        if (Ap && Bp) {
            for (int m0 = 0; m0 < M; m0 += MC) {
                int mc = std::min(MC, M - m0);
                packA_int8(Ap, A, m0, mc, K, Kup, MCblk);
                gemm_mblock_int8_stream(m0, M, N, K, A, B, C, MC, NC, Kup, MCblk, NCblk, Ap, Bp);
            }
        }
        free(Ap); free(Bp);
        return;
    }

    /* ---- 二维 (M×N) 瓦片并行: A、B 各仅全 K 打包一次 (Phase-1 并行) ---- */
    int mb = (M + MC - 1) / MC;
    int nb = (N + NC - 1) / NC;
    size_t Astride = (size_t)Kup / KG * MCblk * CHUNK;
    size_t Bstride = (size_t)Kup / KG * NCblk * CHUNK;
    uint8_t* Apool = (uint8_t*)malloc(mb * Astride + 16);
    int8_t*  Bpool = (int8_t*) malloc(nb * Bstride + 16);
    if (!Apool || !Bpool) { free(Apool); free(Bpool); return; }

    #pragma omp parallel num_threads(nt)
    {
        #pragma omp for schedule(dynamic, 1) nowait
        for (int mid = 0; mid < mb; ++mid) {
            int m0 = mid * MC;
            int mc = std::min(MC, M - m0);
            packA_int8(Apool + (size_t)mid * Astride, A, m0, mc, K, Kup, MCblk);
        }
        #pragma omp for schedule(dynamic, 1) nowait
        for (int nid = 0; nid < nb; ++nid) {
            int n0 = nid * NC;
            int nc = std::min(NC, N - n0);
            packB_int8(Bpool + (size_t)nid * Bstride, B, n0, nc, K, N, Kup, NCblk);
        }
        #pragma omp barrier

        #pragma omp for schedule(dynamic, 1)
        for (int t = 0; t < mb * nb; ++t) {
            int mid = t / nb;
            int nid = t % nb;
            int m0 = mid * MC;
            int n0 = nid * NC;
            run_block_int8(m0, n0, M, N, K, A, B, C, MC, NC, Kup, MCblk, NCblk,
                           Apool + (size_t)mid * Astride,
                           Bpool + (size_t)nid * Bstride);
        }
    }
    free(Apool); free(Bpool);
}
