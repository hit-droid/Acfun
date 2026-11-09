/* Acfun Winograd 卷积: F(4,3) —— 3x3 / stride1 / pad1 的专用快速算法。
 *
 * 这是 MNN 在 Conv 上取得极致吞吐的核心手段之一 (更大 Winograd 瓦片)。相对 F(2,3):
 *   - 每 4x4 输出块 (依赖 6x6 输入块 + 3x3 权) 的乘法量从 16·Cin·Cout·4 (=F(2,3) 每 2x2 块)
 *     降到 36·Cin·Cout, 但每个瓦片产出 4x4=16 输出 (vs F(2,3) 的 4 输出) —— 故每输出 MAC
 *     从 16/4=4 降到 36/16=2.25 (≈ F(2,3) 的 0.56x), 且瓦片数降为 1/4, 大幅摊薄每瓦片的
 *     变换开销与 g_hat 重取 (内存带宽受限卷积的关键)。这正是 MNN 用大瓦片逼近吞吐的路线。
 *
 * 变换矩阵 (Lavin / CVPR-2025 构造, 已数值校验至机器精度):
 *   BT43 (输入变换 6x6):   G43 (滤波变换 6x3):        AT43 (输出变换 4x6):
 *     [ 4  0 -5  0  1  0]    [ 1/4   0     0   ]        [ 1  1  1  1  1  0]
 *     [ 0 -4 -4  1  1  0]    [-1/6 -1/6  -1/6 ]        [ 0  1 -1  2 -2  0]
 *     [ 0  4 -4 -1  1  0]    [-1/6  1/6  -1/6 ]        [ 0  1  1  4  4  0]
 *     [ 0 -2 -1  2  1  0]    [ 1/24 1/12  1/6 ]        [ 0  1 -1  8 -8  1]
 *     [ 0  2 -1 -2  1  0]    [ 1/24 -1/12 1/6 ]
 *     [ 0  4  0 -5  0  1]    [ 0    0     1   ]
 *   2D 可分离应用: d_hat = BT·d·BTᵀ (6x6),  g_hat = G·g·Gᵀ (3x3->6x6),  Y = AT·M·ATᵀ,
 *   其中 M = d_hat ⊙ g_hat 跨通道累加。输出 4x4 块写回时按 OH/OW 裁剪;
 *   输入块起点为 P[4·ty], P[4·tx] (P 为 1 像素 0 填充缓冲)。
 *
 * 实现以通用 -O3 编译, 变换为标量/加法, 跨通道批量累加由编译器向量化; 最热的
 * 累加内层由 winograd43_avx512.cpp (-mavx512f -mfma) 运行时派发做 512-bit 向量化。 */
#include "acfun.h"
#include "internal.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

/* F(4,3) 变换矩阵 (已校验) */
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

/* 单通道 6x6 输入块 -> 36 个 Winograd 域值 (d_hat = BT·d·BTᵀ, 可分离)。
 * d[36] 为 row-major 6x6; dh[36] 为列主序 (dh[i*6+j] = 第 (i,j) 个变换域元素)。 */
inline void input_transform(const float* d, float* dh) {
    /* tmp[i][b] = Σ_a BT43[i][a]·d[a][b] (沿行作用) */
    float tmp[36];
    for (int i = 0; i < 6; ++i) {
        for (int b = 0; b < 6; ++b) {
            float s = 0.f;
            for (int a = 0; a < 6; ++a) s += BT43[i][a] * d[a * 6 + b];
            tmp[i * 6 + b] = s;
        }
    }
    /* dh[i][j] = Σ_b tmp[i][b]·BT43[j][b] (沿列作用) */
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            float s = 0.f;
            for (int b = 0; b < 6; ++b) s += tmp[i * 6 + b] * BT43[j][b];
            dh[i * 6 + j] = s;
        }
    }
}

/* 单通道 3x3 权重 -> 36 个 Winograd 域值 (g_hat = G·g·Gᵀ, 可分离)。
 * g[9] 为 row-major 3x3; gh[36] 为列主序 (gh[i*6+j])。 */
inline void filter_transform(const float* g, float* gh) {
    float tmp[18];   /* i*3 + b */
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

/* 单瓦片计算 (4x4 输出块): 使用调用方提供的线程私有 m_hat(36*Cout)。
 * g_hat(仅依赖权重, 调用内只读) 与 pad(逐图串行填充后只读) 跨线程共享。 */
static void wino43_tile_scalar(int ty, int tx, int OH, int OW, int Cin, int Cout, int act,
                               const float* bias,
                               const float* g_hat, const float* pbase, int pstride,
                               float* m_hat, float* output, int n) {
    int oh0 = 4 * ty, ow0 = 4 * tx;
    std::memset(m_hat, 0, (size_t)36 * Cout * sizeof(float));

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
        float dh[36];
        input_transform(dblock, dh);

        /* m_hat[ij][c] += dh[ij] * g_hat[ij][d][c] (c 连续) */
        for (int ij = 0; ij < 36; ++ij) {
            const float* gh = &g_hat[((size_t)ij * Cin + d) * Cout];
            float*       mh = &m_hat[(size_t)ij * Cout];
            float        v  = dh[ij];
            for (int c = 0; c < Cout; ++c) mh[c] += v * gh[c];
        }
    }

    /* 输出变换 Y = AT·M·ATᵀ, 写回 4x4 输出并融合 bias/ReLU (无分支 vmax) */
    float* yo = output + (size_t)n * OH * OW * Cout;
    for (int c = 0; c < Cout; ++c) {
        /* 收集本通道 m_hat[ij] (ij = i*6+j) 到 M[36], 再 Y[a][b]=Σ_{i,j} AT[a][i]·M[i*6+j]·AT[b][j] */
        float M[36];
        for (int ij = 0; ij < 36; ++ij) M[ij] = m_hat[(size_t)ij * Cout + c];
        float Y[16];
        for (int a = 0; a < 4; ++a) {
            for (int b = 0; b < 4; ++b) {
                float s = 0.f;
                for (int i = 0; i < 6; ++i) {
                    float ri = AT43[a][i];
                    if (ri == 0.f) continue;
                    for (int j = 0; j < 6; ++j) {
                        float cj = AT43[b][j];
                        if (cj == 0.f) continue;
                        s += ri * cj * M[i * 6 + j];
                    }
                }
                Y[a * 4 + b] = s;
            }
        }
        float bc = bias ? bias[c] : 0.0f;
        for (int r = 0; r < 4; ++r) {
            int oh = oh0 + r;
            if (oh >= OH) continue;
            for (int cc = 0; cc < 4; ++cc) {
                int ow = ow0 + cc;
                if (ow >= OW) continue;
                float v = Y[r * 4 + cc] + bc;
                if (act == 1) v = v < 0.0f ? 0.0f : v;
                yo[(size_t)(oh * OW + ow) * Cout + c] = v;
            }
        }
    }
}

/* Winograd F(4,3) 卷积主体: 3x3 / stride1 / pad1, NHWC。要求 H>=3 && W>=3; act=1 融合 ReLU。
 * 权重布局: w[KH*KW*Cin*Cout]。与 F(2,3) 同算法、同布局, 仅瓦片 4x4、变换 6x6。 */
void acfun_winograd43_conv3x3_scalar(const float* input, int N, int H, int W, int Cin,
                                     const float* weight, const float* bias,
                                     int Cout, float* output, int act) {
    const int OH = H, OW = W;                 /* s1 p1 => 输出同尺寸 */
    const int tH = (OH + 3) / 4;              /* 4x4 输出瓦片数 (向上取整) */
    const int tW = (OW + 3) / 4;

    /* 预计算变换后权重 g_hat[36][Cin][Cout], 内层 c 连续。布局: ((ij*Cin + d) * Cout) + c。 */
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

    /* 带 1 像素 0 填充的输入缓冲 (尺寸 (H+2)x(W+2)xCin), 边界瓦片越界读取自然为 0。 */
    std::vector<float> pad((size_t)(H + 2) * (W + 2) * Cin);
    const int pstride = (W + 2) * Cin;

    const long long work = (long long)Cin * Cout * OH * OW;
    const int PARALLEL = (work >= 8000000LL);
    int nth = acfun_effective_cores();
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
                #pragma omp for collapse(2) schedule(static)
                for (int ty = 0; ty < tH; ++ty)
                    for (int tx = 0; tx < tW; ++tx)
                        wino43_tile_scalar(ty, tx, OH, OW, Cin, Cout, act, bias,
                                          g_hat.data(), pbase, pstride,
                                          m_hat.data(), output, n);
            }
        } else {
            std::vector<float> m_hat((size_t)36 * Cout);
            for (int ty = 0; ty < tH; ++ty)
                for (int tx = 0; tx < tW; ++tx)
                    wino43_tile_scalar(ty, tx, OH, OW, Cin, Cout, act, bias,
                                      g_hat.data(), pbase, pstride,
                                      m_hat.data(), output, n);
        }
    }
}

/* 运行时派发: 优先 AVX-512F (512-bit FMA, 寄存器驻留累加), 否则标量 (SSE) 回退。
 * winograd43.cpp 始终以通用指令集编译, 保证可移植; 向量化强度由 winograd43_avx512.cpp 提供。 */
void acfun_winograd43_conv3x3(const float* input, int N, int H, int W, int Cin,
                              const float* weight, const float* bias,
                              int Cout, float* output, int act) {
#if defined(__x86_64__) || defined(_M_X64)
    int b = acfun_backend();
    if (b == 1) {                       /* AVX-512F */
        acfun_winograd43_conv3x3_avx512(input, N, H, W, Cin, weight, bias, Cout, output, act);
        return;
    }
#endif
    acfun_winograd43_conv3x3_scalar(input, N, H, W, Cin, weight, bias, Cout, output, act);
}
