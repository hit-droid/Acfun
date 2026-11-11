/* Acfun INT8 Winograd 卷积: F(2,3) —— 3x3 / stride1 / pad1 的专用快速算法 (对标 MNN int8 路径)
 *
 * 这是 MNN 在 x86 端拿 int8 吞吐的核心手段。思路与 float Winograd 同构, 但全程在
 * 整数域完成, 配合逐张量激活量化 (uint8, 零点 zp=128) 与逐通道权重量化 (int8), 由
 * 量化卷积层 (conv_int8.cpp) 在 3x3/s1/p1 且通道>=4 时自动路由到本函数。
 *
 * 关键工程点 (为何能比肩 MNN int8 卷积):
 *   - 乘法量从 9·Cin·Cout·4 降到 16·Cin·Cout (2.25x), 与 float Winograd 同构;
 *   - 主乘累加 M = d_hat ⊙ g_hat 为 int16×int16→int32, 带宽仅 int16 (float 的 1/2),
 *     再经反量化 (× s_in × 每通道 scale) 还原 float 输出, 量化误差与 im2col+INT8 GEMM 同量级;
 *   - 完全跳过 im2col 缓冲, Winograd 直接以 4x4 输入块做整数变换, 算术强度更高。
 *
 * 整数变换推导 (全部线性, 无新增乘法):
 *   输入 d_hat = BT·d·BTᵀ (BT 整数), 权重 g_hat = G·g·Gᵀ (G 含 0.5)。为消除 0.5 引入小数,
 *   令 G̃ = 2G (整数), 则 g_hat2 = round(G̃·g·G̃ᵀ) = 4·(G·g·Gᵀ) 的整数近似, 携带 ×4 缩放。
 *   逐通道累加后每 (ij,o) 的 4·M 值 T4[ij][o] = Σ_c dh_q[ij][c]·g_hat2[ij][c][o]
 *                              − zp·Q[ij]·Sgh[ij][o]
 *   其中 Q = BT·J·BTᵀ (J 全 1, 常量, 仅 Q[5]=4) 为激活零点校正, Sgh[ij][o]=Σ_c g_hat2[ij][c][o] 为
 *   滤波零点校正。由 Winograd 线性性, 输出 Y = (s_in·s_w[o]/4)·(AT·T4·ATᵀ) + bias (o 为输出通道),
 *   与同口径 float 卷积在「int8 量化」意义下严格等价 (仅 g_hat2 的 ×4 取整引入微小额外误差)。
 *   输出变换以 int64 累加, 避免大通道下 int32 溢出。
 *
 * 性能设计 (与 float Winograd-AVX512 同构的寄存器驻留 + 输出分块, 但用 vpmaddwd 融合乘加):
 *   - 输出通道 o 以 16 路分块 (Cout 向上补齐到 16 倍数 Cout_up); 每个 o 块内以 16 个 zmm 累加器
 *     (acc[ij]) 跨全部输入通道 c 累加, 仅末了写回 T4 一次 (消除旧实现每 (ij,o) 重读 dh[] 的冗余访存)。
 *   - 核心乘累加 T4[ij][o]=Σ_c dh[c]·g[c][o] 是沿 c 的「点积」。AVX-512 的 vpmaddwd (madd_epi16,
 *     512-bit) 单指令完成 a0·b0+a1·b1+… (16 路 int16×int16→int32 融合乘加), 与 float 单 FMA 同量级
 *     指令数。为此 g_hat2 采用交错布局 [ij][pair][2·o+(c&1)]: 一次 512-bit load 即取回某 (c,c+1)
 *     对在 16 个 o 上的交错值 (g[c][o], g[c+1][o]), 以 (dh[c],dh[c+1]) 打包广播作乘子, vpmaddwd 即
 *     产出该对的 16 路 o 累加 —— 单指令完成 2 个 c, 较逐 c 的「符号扩展+vpmulld」省约 2.5x 指令, 且
 *     数据仍是 int16 (带宽仅 float 之半)。奇数 Cin 余 1 个 c 时乘子高 16 置 0 单独处理。
 *   - 输出变换 Y=(s_in·s_w[o]/4)·(AT·T4·ATᵀ) 为本版重点提速对象: 沿用 v0.6.0 float 路线, 改为
 *     512-bit scatter-add (4 个输出元素各持 16 路 int64 累加器, 对 16 个变换域元素按预计算符号表
 *     做 vpsub/vpadd), 消除标量 16×4×4 输出的 ILP 受限与零系数分支 —— 这是 INT8 卷积从「慢于 float」
 *     逼近「比肩 float」的关键。
 *   - 输入量化融合: 此前 conv_int8 需先对整张输入做一遍 lroundf 量化 (inQ), 再由本函数 memcpy 进
 *     填充缓冲, 等于两遍扫描输入。本版改由本函数在填充缓冲时 **就地量化** (AVX-512 向量化、
 *     round-half-away 与 lroundf 逐位一致), 彻底省去整张输入的独立量化遍与 inQ 中间缓冲。
 *   - 跨输出瓦片 (ty,tx) 做 OpenMP 并行, g_hat2/Sgh (权重相关, 只读) 与 pad (逐图填充后只读)
 *     跨线程共享。本 TU 编译带 -mavx512f -mavx512bw -mfma, 由 conv_int8 在检测到 avx512f 时路由。 */
#include "acfun.h"
#include "internal.h"
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <cstdint>

namespace {

/* 单通道 4x4 uint8 输入块 -> 16 个 Winograd 域 int16 值 (d_hat = BT·d·BTᵀ, 可分离)。
 * 布局 dh[ij*Cin + c]: 同一 ij 的 c 连续。 */
inline void input_transform_u8(const uint8_t* d, int16_t* dh, int Cin, int cbase) {
    int16_t U[4][4];
    for (int l = 0; l < 4; ++l) {
        int16_t c0 = (int16_t)d[l], c1 = (int16_t)d[4 + l], c2 = (int16_t)d[8 + l], c3 = (int16_t)d[12 + l];
        U[0][l] = c0 - c2;
        U[1][l] = c1 + c2;
        U[2][l] = (int16_t)(-c1 + c2);
        U[3][l] = (int16_t)(-c1 + c3);
    }
    for (int i = 0; i < 4; ++i) {
        int16_t u0 = U[i][0], u1 = U[i][1], u2 = U[i][2], u3 = U[i][3];
        dh[(size_t)(i * 4 + 0) * Cin + cbase] = u0 - u2;
        dh[(size_t)(i * 4 + 1) * Cin + cbase] = u1 + u2;
        dh[(size_t)(i * 4 + 2) * Cin + cbase] = (int16_t)(-u1 + u2);
        dh[(size_t)(i * 4 + 3) * Cin + cbase] = (int16_t)(-u1 + u3);
    }
}

/* 单通道 3x3 int8 权重 -> 16 个 int16 滤波变换值 (g_hat2 = round(G̃·g·G̃ᵀ), 携带 ×4 缩放)。
 * g[9] row-major (x*3+y); gh2[16] = g_hat2[ij]。G̃ = 2G = [[2,0,0],[1,1,1],[1,-1,1],[0,0,2]]。 */
inline void filter_transform_int8(const int8_t* g, int16_t* gh2) {
    int16_t T[4][3];
    for (int y = 0; y < 3; ++y) {
        int16_t a = (int16_t)g[y], b = (int16_t)g[3 + y], c = (int16_t)g[6 + y];
        T[0][y] = (int16_t)(2 * a);
        T[1][y] = (int16_t)(a + b + c);
        T[2][y] = (int16_t)(a - b + c);
        T[3][y] = (int16_t)(2 * c);
    }
    for (int i = 0; i < 4; ++i) {
        int16_t t0 = T[i][0], t1 = T[i][1], t2 = T[i][2];
        gh2[i * 4 + 0] = (int16_t)(2 * t0);
        gh2[i * 4 + 1] = (int16_t)(t0 + t1 + t2);
        gh2[i * 4 + 2] = (int16_t)(t0 - t1 + t2);
        gh2[i * 4 + 3] = (int16_t)(2 * t2);
    }
}

/* 每线程私有 scratch: dhq[16*Cin] (本瓦片输入变换) 与 T4[16*Cout_up] (逐瓦片累加), 复用避免逐瓦片分配。 */
static int16_t* dhq_scratch(int Cin) {
    static thread_local std::vector<int16_t> buf;
    if ((int)buf.size() < 16 * Cin) buf.resize((size_t)16 * Cin);
    return buf.data();
}
static int32_t* t4_scratch(int Cout_up) {
    static thread_local std::vector<int32_t> buf;
    if ((int)buf.size() < 16 * Cout_up) buf.resize((size_t)16 * Cout_up);
    return buf.data();
}

/* 输出变换符号表: SIGN[e][ij] = AT[r][i]·AT[c][j] (r=e/2, c=e%2), 仅取 ±1 与 0。
 * e=0..3 对应输出元素 (r,c)∈{(0,0),(0,1),(1,0),(1,1)}; ij=i*4+j。与标量输出变换逐位等价。 */
static const int8_t (&int8_wino_signs())[4][16] {
    static int8_t S[4][16];
    static bool done = false;
    if (!done) {
        static const int AT[2][4] = {{1, 1, 1, 0}, {0, 1, -1, 1}};
        for (int e = 0; e < 4; ++e) {
            int r = e >> 1, c = e & 1;
            for (int ij = 0; ij < 16; ++ij) {
                int i = ij >> 2, j = ij & 3;
                S[e][ij] = (int8_t)(AT[r][i] * AT[c][j]);
            }
        }
        done = true;
    }
    return S;
}

/* 单 o 块 (16 路) 输出变换: Y = (s_in·s_w[o]/4)·(AT·T4·ATᵀ) + bias, int64 累加防溢出。
 * 4 个输出元素各持 16 路 int64 累加器 (2 zmm), 对 16 个变换域元素按符号表 vpsub/vpadd, 末了
 * int64→float 乘 wso 加 bias (+ReLU) 写回。wscale/bias 已补齐到 Cout_up (避免越界读), 仅在
 * o<Cout 的通道写回 (掩码)。 */
inline void wino_output_block_int8(const int32_t* T4, int Cout_up, int ob, int Cout,
                                   const float* wscale, const float* bias, int act,
                                   float s_in, float* yo, int OH, int OW,
                                   int oh0, int ow0) {
    static const int8_t (&S)[4][16] = int8_wino_signs();
    __m512i acc[4][2];
    for (int e = 0; e < 4; ++e) {
        acc[e][0] = _mm512_setzero_si512();
        acc[e][1] = _mm512_setzero_si512();
    }
    for (int ij = 0; ij < 16; ++ij) {
        __m512i t32 = _mm512_loadu_si512((const __m512i*)&T4[(size_t)ij * Cout_up + ob]);
        __m256i lo32 = _mm512_castsi512_si256(t32);
        __m256i hi32 = _mm512_extracti32x8_epi32(t32, 1);
        __m512i lo64 = _mm512_cvtepi32_epi64(lo32);   /* 通道 o..o+7 */
        __m512i hi64 = _mm512_cvtepi32_epi64(hi32);   /* 通道 o+8..o+15 */
        for (int e = 0; e < 4; ++e) {
            int8_t sgn = S[e][ij];
            if (sgn > 0) {
                acc[e][0] = _mm512_add_epi64(acc[e][0], lo64);
                acc[e][1] = _mm512_add_epi64(acc[e][1], hi64);
            } else if (sgn < 0) {
                acc[e][0] = _mm512_sub_epi64(acc[e][0], lo64);
                acc[e][1] = _mm512_sub_epi64(acc[e][1], hi64);
            }
        }
    }
    __m512 wsc = _mm512_loadu_ps(&wscale[ob]);
    __m512 wso = _mm512_mul_ps(_mm512_set1_ps(s_in * 0.25f), wsc);
    __m512 bv  = bias ? _mm512_loadu_ps(&bias[ob]) : _mm512_setzero_ps();
    int nvalid = Cout - ob; if (nvalid > 16) nvalid = 16; if (nvalid < 0) nvalid = 0;
    __mmask16 fullmask = (__mmask16)((nvalid >= 16) ? 0xFFFFu : ((1u << nvalid) - 1u));
    for (int e = 0; e < 4; ++e) {
        int oh = oh0 + (e >> 1), ow = ow0 + (e & 1);
        if (oh >= OH || ow >= OW) continue;
        /* acc[e][0/1] 各 8 个 int64, 分别转 8 个 float 后合并为 16 路 vf (低 8=o..o+7, 高 8=o+8..o+15) */
        __m256 f0 = _mm512_cvtepi64_ps(acc[e][0]);
        __m256 f1 = _mm512_cvtepi64_ps(acc[e][1]);
        __m512 vf = _mm512_insertf32x8(_mm512_castps256_ps512(f0), f1, 1);
        vf = _mm512_fmadd_ps(vf, wso, bv);
        if (act == 1) vf = _mm512_max_ps(vf, _mm512_setzero_ps());
        float* dst = yo + (size_t)(oh * OW + ow) * Cout + ob;
        _mm512_mask_storeu_ps(dst, fullmask, vf);
    }
}

/* 单瓦片计算 (2x2 输出块)。g_hat2/Sgh (只读, 权重相关, 交错布局) 与 pad (只读) 跨线程共享;
 * dhq/T4 为线程私有 scratch。输入变换在瓦片内联完成, 使 d_hat 驻留 L1 (缓存友好)。 */
static void wino_tile_int8(int ty, int tx, int OH, int OW, int Cin, int Cout, int Cout_up, int num_pairs, int act,
                           const float* bias, const float* wscale,
                           const int16_t* g_hat2, const int32_t* Sgh,
                           const uint8_t* pbase, int pstride, float s_in,
                           int32_t* T4, float* output, int n) {
    const int ZP = 128;
    int oh0 = 2 * ty, ow0 = 2 * tx;

    /* 1) 本瓦片所有输入通道的 d_hat (int16), 布局 dh[ij*Cin + c] (ij 内 c 连续) —— 瓦片内联, L1 驻留 */
    int16_t* dh = dhq_scratch(Cin);
    for (int c = 0; c < Cin; ++c) {
        int pr0 = 2 * ty, pc0 = 2 * tx;
        uint8_t dblk[16];
        for (int k = 0; k < 4; ++k) {
            int pr = pr0 + k;
            const uint8_t* prow = (pr >= 0 && pr < OH + 2) ? &pbase[(size_t)pr * pstride] : nullptr;
            for (int l = 0; l < 4; ++l) {
                int pc = pc0 + l;
                if (prow && pc >= 0 && pc < OW + 2) dblk[k * 4 + l] = prow[(size_t)pc * Cin + c];
                else                                 dblk[k * 4 + l] = (uint8_t)ZP;   /* 越界 = 零点 zp */
            }
        }
        input_transform_u8(dblk, dh, Cin, c);
    }

    /* 2) 输出通道 16 路分块; 每个 o 块内以 16 个 zmm 累加器 (acc[ij]) 跨全部输入通道 c 做点积累加,
     *    末了写回 T4 一次。对固定 (ij,o), T4=Σ_c dh[c]·g[c][o] 是沿 c 的点积, 用 vpmaddwd 以 (c,c+1)
     *    为一组融合乘加: g_hat2 交错布局 [ij][pair][2·o+(c&1)] 使一次 512-bit load 取回 16 个 o 的
     *    (g[c][o],g[c+1][o]) 交错, 乘子 A 以 (dh[c],dh[c+1]) 打包广播, vpmaddwd 即产出该组 16 路 o 累加。 */
    const int32_t qcorr5 = (int32_t)ZP * 4;   /* ij==5 时的激活零点校正系数 zp·Q[5] (Q[5]=4) */
    for (int ob = 0; ob < Cout_up; ob += 16) {
        for (int ij = 0; ij < 16; ++ij) {
            __m512i acc = _mm512_setzero_si512();
            int c = 0;
            for (; c + 1 < Cin; c += 2) {                       /* vpmaddwd 成对处理 (c, c+1) */
                const int16_t* gp = &g_hat2[((size_t)(ij * num_pairs + (c >> 1)) * 2 * Cout_up) + (2 * ob)];
                __m512i B = _mm512_loadu_si512((const __m512i*)gp);
                int16_t dhc  = dh[(size_t)ij * Cin + c];
                int16_t dhc1 = dh[(size_t)ij * Cin + c + 1];
                int32_t ap = ((int32_t)dhc1 << 16) | (uint16_t)dhc;   /* (dh[c+1], dh[c]) 打包 */
                __m512i A = _mm512_set1_epi32(ap);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(A, B));
            }
            if (c < Cin) {                                      /* 奇数 Cin 余 1 个 c: 乘子高 16 置 0 */
                const int16_t* gp = &g_hat2[((size_t)(ij * num_pairs + (c >> 1)) * 2 * Cout_up) + (2 * ob)];
                __m512i B = _mm512_loadu_si512((const __m512i*)gp);
                int16_t dhc = dh[(size_t)ij * Cin + c];
                int32_t ap = (uint16_t)dhc;                 /* 高 16 = 0 => vpmaddwd 仅留 dhc·g[c][o] */
                __m512i A = _mm512_set1_epi32(ap);
                acc = _mm512_add_epi32(acc, _mm512_madd_epi16(A, B));
            }
            if (ij == 5) {
                __m512i sg = _mm512_loadu_si512((const __m512i*)&Sgh[(size_t)ij * Cout_up + ob]);
                acc = _mm512_sub_epi32(acc, _mm512_mullo_epi32(_mm512_set1_epi32(qcorr5), sg));
            }
            _mm512_storeu_si512((__m512i*)&T4[(size_t)ij * Cout_up + ob], acc);
        }

        /* 3) 512-bit 向量化输出变换 (scatter-add, int64 累加) —— 本版提速重点 */
        float* yo = output + (size_t)n * OH * OW * Cout;   /* 按 batch 偏移到本图输出基址 */
        wino_output_block_int8(T4, Cout_up, ob, Cout, wscale, bias, act, s_in, yo, OH, OW, oh0, ow0);
    }
}

} /* namespace */

/* INT8 Winograd F(2,3) 卷积主体: 3x3 / stride1 / pad1, NHWC, float 输入 (本函数内就地量化, 零点 zp=128)。
 * 权重 wq 为逐通道量化 int8 (布局 [KH*KW*Cin * Cout], 即 [((k*Cin+c)*Cout)+o])。
 * inv_s = 127/max|x| (与 conv_int8 量化步长严格一致, 故 x_q = round(x·inv_s)+ZP)。
 * 要求 H>=3 && W>=3 && Cin>=4 && Cout>=4; act=1 融合 ReLU。bias 为 [Cout], 可为 NULL。 */
void acfun_winograd_conv3x3_int8(const float* input, int N, int H, int W, int Cin,
                                 const int8_t* wq, const float* wscale, const float* bias, int Cout,
                                 float s_in, float inv_s, float* output, int act) {
    const int OH = H, OW = W;                  /* s1 p1 => 输出同尺寸 */
    const int tH = (OH + 1) / 2;               /* 2x2 输出瓦片数 (向上取整) */
    const int tW = (OW + 1) / 2;
    const int Cout_up = ((Cout + 15) / 16) * 16;   /* 输出通道向上补齐到 16 倍数, 供 512-bit 分块 */
    const int num_pairs = (Cin + 1) / 2;           /* (c,c+1) 对数, 适配 vpmaddwd 成对点积 */

    /* 预计算滤波变换 g_hat2[16][num_pairs][2·Cout_up] (交错布局 [ij][pair][2·o+(c&1)]) 与
     * 零点校正 Sgh[16][Cout_up]。交错布局使某 (c,c+1) 对在 16 个 o 上的值连续可 512-bit 加载。
     * 补齐区 (o∈[Cout,Cout_up)) 保持 0, 不会被输出变换读取。 */
    std::vector<int16_t> g_hat2((size_t)16 * num_pairs * 2 * Cout_up, 0);
    std::vector<int32_t> Sgh((size_t)16 * Cout_up, 0);
    for (int c = 0; c < Cin; ++c) {
        for (int o = 0; o < Cout; ++o) {
            int8_t g[9];
            for (int k = 0; k < 9; ++k) g[k] = wq[((size_t)k * Cin + c) * Cout + o];
            int16_t gh2[16];
            filter_transform_int8(g, gh2);
            for (int ij = 0; ij < 16; ++ij) {
                size_t idx = ((size_t)(ij * num_pairs + (c >> 1)) * 2 * Cout_up) + (size_t)(2 * o + (c & 1));
                g_hat2[idx] = gh2[ij];
                Sgh[(size_t)ij * Cout_up + o] += (int32_t)gh2[ij];
            }
        }
    }

    /* wscale/bias 补齐到 Cout_up, 供向量化输出变换按 16 路整块读取 (避免 o>=Cout 越界读); 输出
     * 仅写回 o<Cout 通道 (掩码), 补齐区恒为 0 不会被读。 */
    std::vector<float> wscale_up(Cout_up, 0.0f), bias_up(Cout_up, 0.0f);
    for (int o = 0; o < Cout; ++o) { wscale_up[o] = wscale[o]; bias_up[o] = bias ? bias[o] : 0.0f; }

    /* 带 1 像素 0 填充的输入缓冲 (尺寸 (H+2)x(W+2)xCin); 边界瓦片越界读取自然为零点 zp=128 —— 反量化
     * 后恰好还原 float 0, 否则边缘会被错读成 -s_in·128 而污染全部输出 (此前数值错误的根因)。 */
    std::vector<uint8_t> pad((size_t)(H + 2) * (W + 2) * Cin);
    const int pstride = (W + 2) * Cin;
    const int ZP = 128;
    const __m512i vzp = _mm512_set1_epi32(ZP);
    const __m512 vinv = _mm512_set1_ps(inv_s);
    const __m512 vhalf = _mm512_set1_ps(0.5f);

    const long long work = (long long)Cin * Cout * OH * OW;
    const int PARALLEL = (work >= 8000000LL);
    int nth = acfun_effective_cores();
    if (nth > tH * tW) nth = tH * tW;
    if (nth < 1) nth = 1;

    for (int n = 0; n < N; ++n) {
        const float* x = input + (size_t)n * H * W * Cin;
        /* 零填充在 uint8 域必须为零点 zp (=128): 反量化后恰好还原 float 0。 */
        std::memset(pad.data(), (uint8_t)ZP, pad.size() * sizeof(uint8_t));
        for (int ih = 0; ih < H; ++ih)
            for (int iw = 0; iw < W; ++iw) {
                uint8_t* dst = &pad[((size_t)(ih + 1) * (W + 2) + (iw + 1)) * Cin];
                const float* xrow = x + ((size_t)ih * W + iw) * Cin;
                int c = 0;
                /* 就地量化 (向量化, round-half-away 与 lroundf 逐位一致):
                 * t = x·inv_s; r = t + 0.5·sign(t); ri = trunc(r) = round-half-away(t); 再 +ZP 并 clamp 到 [0,255]。
                 * 注意: 必须先用 128-bit packus 把 ri16 的「低 128 位 + 高 128 位」打包成 16 个 uint8,
                 * 不能用 _mm256_packus_epi16(ri16,ri16)+cast(低128): 那样会丢弃 ri[8..15]、且把 ri[0..7] 重复两遍。 */
                for (; c + 16 <= Cin; c += 16) {
                    __m512 xv = _mm512_loadu_ps(xrow + c);
                    __m512 t  = _mm512_mul_ps(xv, vinv);                 /* t = x·inv_s */
                    __m512 sgn = _mm512_castsi512_ps(
                        _mm512_and_si512(_mm512_castps_si512(t), _mm512_set1_epi32(0x80000000u)));
                    __m512 r = _mm512_add_ps(t, _mm512_or_ps(vhalf, sgn)); /* t + 0.5·sign(t) → 半数远离取整 */
                    __m512i ri = _mm512_cvttps_epi32(r);                  /* round-half-away(t) (与 lroundf 逐位一致) */
                    ri = _mm512_add_epi32(ri, vzp);                       /* + 零点 ZP (整数域, 避免偏移破坏截断恒等式) */
                    ri = _mm512_max_epi32(ri, _mm512_setzero_si512());
                    ri = _mm512_min_epi32(ri, _mm512_set1_epi32(255));
                    __m256i ri16 = _mm512_cvtepi32_epi16(ri);            /* 低 16 个 int32 → 16 int16 */
                    __m128i ri8  = _mm_packus_epi16(_mm256_castsi256_si128(ri16),
                                                    _mm256_extracti128_si256(ri16, 1));  /* 16 uint8 (低128+高128) */
                    _mm_storeu_si128((__m128i*)(dst + c), ri8);
                }
                for (; c < Cin; ++c) {                                   /* 尾通道标量兜底 */
                    int v = (int)std::lroundf(xrow[c] * inv_s) + ZP;
                    if (v < 0) v = 0; else if (v > 255) v = 255;
                    dst[c] = (uint8_t)v;
                }
            }
        const uint8_t* pbase = pad.data();

        /* 逐瓦片计算: 输入变换在瓦片内联完成 (d_hat 驻留 L1), 输出直接写回。 */
        if (PARALLEL) {
            #pragma omp parallel num_threads(nth)
            {
                std::vector<int32_t> T4((size_t)16 * Cout_up);   /* 线程私有逐瓦片累加缓冲 */
                #pragma omp for collapse(2) schedule(static)
                for (int ty = 0; ty < tH; ++ty)
                    for (int tx = 0; tx < tW; ++tx)
                        wino_tile_int8(ty, tx, OH, OW, Cin, Cout, Cout_up, num_pairs, act, bias_up.data(), wscale_up.data(),
                                       g_hat2.data(), Sgh.data(),
                                       pbase, pstride, s_in, T4.data(), output, n);
            }
        } else {
            std::vector<int32_t> T4((size_t)16 * Cout_up);
            for (int ty = 0; ty < tH; ++ty)
                for (int tx = 0; tx < tW; ++tx)
                    wino_tile_int8(ty, tx, OH, OW, Cin, Cout, Cout_up, num_pairs, act, bias_up.data(), wscale_up.data(),
                                   g_hat2.data(), Sgh.data(),
                                   pbase, pstride, s_in, T4.data(), output, n);
        }
    }
}
