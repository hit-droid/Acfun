/* Acfun Winograd 卷积 AVX-512+FMA 变体 (F(4,3), 3x3 / stride1 / pad1)。
 *
 * 与 winograd43.cpp 标量实现同算法 (6x6 变换 / 36 变换域元素), 仅在最热的「累加内层」
 *   m_hat[ij][c] += dh[ij] * g_hat[ij][d][c]
 * 上做三层优化, 与 F(2,3) 的 AVX-512 路线一致:
 *
 *   1) 宽度向量化: 512-bit FMA, 一次 16 通道 (c 连续);
 *   2) 寄存器驻留累加 (突破 32 zmm 限制): 36 个 ij 拆为两组 (0..17 / 18..35) 各 18 个 zmm
 *      累加器, 每组跨全部输入通道 d 累加、仅末了写回 m_hat 一次 —— 彻底消除「每 (d,ij)
 *      重读整条 m_hat[c]」的冗余访存 (F(4,3) 若逐 ij RMW 会引入 ~9K/瓦片 的 m_hat 重读);
 *      dh[d][36] 每通道只算一次并缓存于 dh_all, 瓦片间复用预分配的 m_hat/dh_all/pad 缓冲;
 *   3) 瓦片并行: 4x4 输出瓦片彼此独立 (写入互不相交的输出区), 对 (ty,tx) 网格做 OpenMP 并行;
 *      每线程持私有 m_hat/dh_all, g_hat(仅依赖权重, 调用内只读) 与 pad(逐图串行填充后只读)
 *      跨线程共享。受 cgroup CPU 配额约束, 大卷积最多 ~核心数倍加速; 小卷积回落纯串行。
 *
 * 算术与标量路径完全一致 (同为 fma 舍入序列), 故数值等价; 余数 c 块以 __mmask16 收尾,
 * ReLU 以无分支 vmax 语义融合。编译单元需带 -mavx512f -mfma (见 CMakeLists.txt)。 */
#include "acfun.h"
#include "internal.h"
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

static const float BT43[6][6] = {
    {  4.f,   0.f,  -5.f,   0.f,   1.f,   0.f },
    {  0.f,  -4.f,  -4.f,   1.f,   1.f,   0.f },
    {  0.f,   4.f,  -4.f,  -1.f,   1.f,   0.f },
    {  0.f,  -2.f,  -1.f,   2.f,   1.f,   0.f },
    {  0.f,   2.f,  -1.f,  -2.f,   1.f,   0.f },
    {  0.f,   4.f,   0.f,  -5.f,   0.f,   1.f },
};
static const float G43[6][3] = {
    {  0.25f,       0.f,        0.f   },
    { -1.f/6.f,    -1.f/6.f,   -1.f/6.f },
    { -1.f/6.f,     1.f/6.f,   -1.f/6.f },
    {  1.f/24.f,    1.f/12.f,   1.f/6.f },
    {  1.f/24.f,   -1.f/12.f,   1.f/6.f },
    {  0.f,         0.f,        1.f   },
};
static const float AT43[4][6] = {
    { 1.f,  1.f,  1.f,  1.f,  1.f,  0.f },
    { 0.f,  1.f, -1.f,  2.f, -2.f,  0.f },
    { 0.f,  1.f,  1.f,  4.f,  4.f,  0.f },
    { 0.f,  1.f, -1.f,  8.f, -8.f,  1.f },
};

/* 单通道 6x6 输入块 -> 36 个 Winograd 域值 (d_hat = BT·d·BTᵀ, 可分离)。 */
inline void input_transform(const float* d, float* dh) {
    float tmp[36];
    for (int i = 0; i < 6; ++i) {
        for (int b = 0; b < 6; ++b) {
            float s = 0.f;
            for (int a = 0; a < 6; ++a) s += BT43[i][a] * d[a * 6 + b];
            tmp[i * 6 + b] = s;
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.f;
            for (int b = 0; b < 6; ++b) s += tmp[i * 6 + b] * BT43[j][b];
            dh[i * 6 + j] = s;
        }
    }
}

inline void filter_transform(const float* g, float* gh) {
    float tmp[18];
    for (int i = 0; i < 6; ++i) {
        for (int b = 0; b < 3; ++b) {
            float s = 0.f;
            for (int a = 0; a < 3; ++a) s += G43[i][a] * g[a * 3 + b];
            tmp[i * 3 + b] = s;
        }
    }
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.f;
            for (int b = 0; b < 3; ++b) s += tmp[i * 3 + b] * G43[j][b];
            gh[i * 6 + j] = s;
        }
    }
}

} /* namespace */

/* 单瓦片计算 (4x4 输出块): 使用调用方提供的线程私有 m_hat(36*Cout) 与 dh_all(Cin*36)。
 * g_hat / pad 跨线程只读共享。 */
static void wino43_tile_avx512(int ty, int tx, int OH, int OW, int Cin, int Cout, int act,
                               const float* bias,
                               const float* g_hat, const float* pbase, int pstride,
                               float* m_hat, float* dh_all, float* output, int n) {
    int oh0 = 4 * ty, ow0 = 4 * tx;

    /* 先对每个通道 d 算好输入变换 dh[d][36] (一次提取+变换, 瓦片内复用) */
    for (int d = 0; d < Cin; ++d) {
        int pr0 = 4 * ty, pc0 = 4 * tx;
        float dblock[36];
        for (int k = 0; k < 6; ++k) {
            int pr = pr0 + k;
            const float* prow = (pr >= 0 && pr < OH + 2)
                ? &pbase[(size_t)pr * pstride] : nullptr;
            for (int l = 0; l < 6; ++l) {
                int pc = pc0 + l;
                if (prow && pc >= 0 && pc < OW + 2)
                    dblock[k * 6 + l] = prow[(size_t)pc * Cin + d];
                else
                    dblock[k * 6 + l] = 0.0f;
            }
        }
        input_transform(dblock, &dh_all[(size_t)d * 36]);
    }

    /* c 维 16 路分块; 每个 c 块: 先用 18+18 zmm 寄存器驻留累加算 m_hat[k][c0:c0+16]
     * (跨全部 d 累加, 仅末了写回一次), 立即做向量化输出变换 Y=AT·M·ATᵀ —— 16 个输出空间
     * 位置各持一条 c 块累加器, 对 36 个变换域 k 做 scatter-add (vk=m_hat[k][c:c+16],
     * oacc[p]+=coef[p][k]·vk)。彻底消除原逐通道标量 16×36 输出的 ILP 受限与零系数分支,
     * 这是 F(4,3) 相对 F(2,3) 提速的关键 (F(4,3) 输出变换域 36 元素 vs F(2,3) 16 元素)。 */
    float coef[16][36];
    for (int p = 0; p < 16; ++p) {
        int a = p / 4, b = p % 4;
        for (int k = 0; k < 36; ++k) coef[p][k] = AT43[a][k / 6] * AT43[b][k % 6];
    }
    float* yo = output + (size_t)n * OH * OW * Cout;

    int c = 0;
    for (; c + 16 <= Cout; c += 16) {
        /* --- m_hat 累加 (18+18 zmm, 跨全部 d) --- */
        __m512 accA[18];
        for (int ij = 0; ij < 18; ++ij) accA[ij] = _mm512_setzero_ps();
        for (int d = 0; d < Cin; ++d) {
            const float* dh = &dh_all[(size_t)d * 36];
            for (int ij = 0; ij < 18; ++ij) {
                __m512 g = _mm512_loadu_ps(&g_hat[((size_t)ij * Cin + d) * Cout + c]);
                accA[ij] = _mm512_fmadd_ps(_mm512_set1_ps(dh[ij]), g, accA[ij]);
            }
        }
        for (int ij = 0; ij < 18; ++ij)
            _mm512_storeu_ps(&m_hat[(size_t)ij * Cout + c], accA[ij]);
        __m512 accB[18];
        for (int ij = 0; ij < 18; ++ij) accB[ij] = _mm512_setzero_ps();
        for (int d = 0; d < Cin; ++d) {
            const float* dh = &dh_all[(size_t)d * 36];
            for (int ij = 18; ij < 36; ++ij) {
                __m512 g = _mm512_loadu_ps(&g_hat[((size_t)ij * Cin + d) * Cout + c]);
                accB[ij - 18] = _mm512_fmadd_ps(_mm512_set1_ps(dh[ij]), g, accB[ij - 18]);
            }
        }
        for (int ij = 18; ij < 36; ++ij)
            _mm512_storeu_ps(&m_hat[(size_t)ij * Cout + c], accB[ij - 18]);

        /* --- 向量化输出变换 (16 输出位置 × c 块) --- */
        __m512 oacc[16];
        for (int p = 0; p < 16; ++p) oacc[p] = _mm512_setzero_ps();
        for (int k = 0; k < 36; ++k) {
            __m512 vk = _mm512_loadu_ps(&m_hat[(size_t)k * Cout + c]);
            for (int p = 0; p < 16; ++p)
                oacc[p] = _mm512_fmadd_ps(_mm512_set1_ps(coef[p][k]), vk, oacc[p]);
        }
        for (int p = 0; p < 16; ++p) {
            int a = p / 4, b = p % 4, oh = oh0 + a, ow = ow0 + b;
            if (oh < OH && ow < OW) {
                __m512 v = oacc[p];
                if (bias) v = _mm512_add_ps(v, _mm512_loadu_ps(&bias[c]));
                if (act == 1) v = _mm512_max_ps(v, _mm512_setzero_ps());
                _mm512_storeu_ps(&yo[(size_t)(oh * OW + ow) * Cout + c], v);
            }
        }
    }
    if (c < Cout) {                               /* 余数列块: masking 收尾 */
        int rem = Cout - c;
        __mmask16 msk = (__mmask16)((1u << rem) - 1);
        __m512 accA[18];
        for (int ij = 0; ij < 18; ++ij) accA[ij] = _mm512_setzero_ps();
        for (int d = 0; d < Cin; ++d) {
            const float* dh = &dh_all[(size_t)d * 36];
            for (int ij = 0; ij < 18; ++ij) {
                __m512 g = _mm512_maskz_loadu_ps(msk, &g_hat[((size_t)ij * Cin + d) * Cout + c]);
                accA[ij] = _mm512_fmadd_ps(_mm512_set1_ps(dh[ij]), g, accA[ij]);
            }
        }
        for (int ij = 0; ij < 18; ++ij)
            _mm512_mask_storeu_ps(&m_hat[(size_t)ij * Cout + c], msk, accA[ij]);
        __m512 accB[18];
        for (int ij = 0; ij < 18; ++ij) accB[ij] = _mm512_setzero_ps();
        for (int d = 0; d < Cin; ++d) {
            const float* dh = &dh_all[(size_t)d * 36];
            for (int ij = 18; ij < 36; ++ij) {
                __m512 g = _mm512_maskz_loadu_ps(msk, &g_hat[((size_t)ij * Cin + d) * Cout + c]);
                accB[ij - 18] = _mm512_fmadd_ps(_mm512_set1_ps(dh[ij]), g, accB[ij - 18]);
            }
        }
        for (int ij = 18; ij < 36; ++ij)
            _mm512_mask_storeu_ps(&m_hat[(size_t)ij * Cout + c], msk, accB[ij - 18]);

        __m512 oacc[16];
        for (int p = 0; p < 16; ++p) oacc[p] = _mm512_setzero_ps();
        for (int k = 0; k < 36; ++k) {
            __m512 vk = _mm512_maskz_loadu_ps(msk, &m_hat[(size_t)k * Cout + c]);
            for (int p = 0; p < 16; ++p)
                oacc[p] = _mm512_fmadd_ps(_mm512_set1_ps(coef[p][k]), vk, oacc[p]);
        }
        for (int p = 0; p < 16; ++p) {
            int a = p / 4, b = p % 4, oh = oh0 + a, ow = ow0 + b;
            if (oh < OH && ow < OW) {
                __m512 v = oacc[p];
                if (bias) v = _mm512_add_ps(v, _mm512_maskz_loadu_ps(msk, &bias[c]));
                if (act == 1) v = _mm512_max_ps(v, _mm512_setzero_ps());
                _mm512_mask_storeu_ps(&yo[(size_t)(oh * OW + ow) * Cout + c], msk, v);
            }
        }
    }
}

void acfun_winograd43_conv3x3_avx512(const float* input, int N, int H, int W, int Cin,
                                     const float* weight, const float* bias,
                                     int Cout, float* output, int act) {
    const int OH = H, OW = W;                 /* s1 p1 => 输出同尺寸 */
    const int tH = (OH + 3) / 4;              /* 4x4 输出瓦片数 (向上取整) */
    const int tW = (OW + 3) / 4;

    /* 预计算变换后权重 g_hat[36][Cin][Cout], 内层 c 连续。布局: ((ij*Cin+d)*Cout)+c。 */
    std::vector<float> g_hat((size_t)36 * Cin * Cout);
    for (int d = 0; d < Cin; ++d) {
        for (int c = 0; c < Cout; ++c) {
            float g[9];
            for (int k = 0; k < 9; ++k)
                g[k] = weight[((size_t)k * Cin + d) * Cout + c];
            float gh[36];
            filter_transform(g, gh);
            for (int ij = 0; ij < 36; ++ij)
                g_hat[((size_t)ij * Cin + d) * Cout + c] = gh[ij];
        }
    }

    /* 带 1 像素 0 填充的输入缓冲 (尺寸 (H+2)x(W+2)xCin), 逐图填充后只读共享。 */
    std::vector<float> pad((size_t)(H + 2) * (W + 2) * Cin);
    const int pstride = (W + 2) * Cin;

    const long long work = (long long)Cin * Cout * OH * OW;
    const int PARALLEL = (work >= 8000000LL);
    int nth = acfun_effective_cores();          /* cgroup 感知 */
    if (nth > tH * tW) nth = tH * tW;
    if (nth < 1) nth = 1;

    for (int n = 0; n < N; ++n) {
        const float* x = input + (size_t)n * H * W * Cin;
        std::memset(pad.data(), 0, pad.size() * sizeof(float));
        for (int ih = 0; ih < H; ++ih)
            for (int iw = 0; iw < W; ++iw)
                memcpy(&pad[((size_t)(ih + 1) * (W + 2) + (iw + 1)) * Cin],
                       x + ((size_t)ih * W + iw) * Cin, Cin * sizeof(float));
        const float* pbase = pad.data();

        if (PARALLEL) {
            #pragma omp parallel num_threads(nth)
            {
                std::vector<float> m_hat((size_t)36 * Cout);
                std::vector<float> dh_all((size_t)Cin * 36);
                #pragma omp for collapse(2) schedule(static)
                for (int ty = 0; ty < tH; ++ty)
                    for (int tx = 0; tx < tW; ++tx)
                        wino43_tile_avx512(ty, tx, OH, OW, Cin, Cout, act, bias,
                                           g_hat.data(), pbase, pstride,
                                           m_hat.data(), dh_all.data(), output, n);
            }
        } else {
            std::vector<float> m_hat((size_t)36 * Cout);
            std::vector<float> dh_all((size_t)Cin * 36);
            for (int ty = 0; ty < tH; ++ty)
                for (int tx = 0; tx < tW; ++tx)
                    wino43_tile_avx512(ty, tx, OH, OW, Cin, Cout, act, bias,
                                       g_hat.data(), pbase, pstride,
                                       m_hat.data(), dh_all.data(), output, n);
        }
    }
}
