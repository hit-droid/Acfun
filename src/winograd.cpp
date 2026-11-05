/* Acfun Winograd 卷积: F(2,3) —— 3x3 / stride1 / pad1 的专用快速算法。
 *
 * 这是 MNN 在 Conv 上取得极致吞吐的核心手段之一。对于 3x3 卷积, 朴素的
 * im2col+GEMM 每个输出像素需 9·Cin 次乘加; Winograd 把每 2x2 输出块(依赖
 * 4x4 输入块 + 3x3 权)的乘法量从 9·Cin·Cout·4 降到 16·Cin·Cout —— 乘法次数
 * 减少 2.25x, 且变换仅含加法与 ±1/±2 的廉价缩放。算术强度越高(通道数越大、
 * 特征图越大)收益越显著。
 *
 * 变换矩阵 (Lavin & Gray, Fast Algorithms for CNN; 经 wincnn 校验):
 *   BT (输入变换 4x4):   G (滤波变换 4x3):        AT (输出变换 2x4):
 *     [ 1  0 -1  0]        [ 1   0   0 ]            [ 1  1  1  0]
 *     [ 0  1  1  0]        [.5  .5  .5 ]            [ 0  1 -1  1]
 *     [ 0 -1  1  0]        [.5 -.5  .5 ]
 *     [ 0 -1  0  1]        [ 0   0   1 ]
 *   2D 可分离应用: d_hat = BT d BTᵀ,  g_hat = G g Gᵀ,  Y = AT M ATᵀ,
 *   其中 M = d_hat ⊙ g_hat (逐元素乘, 跨通道累加)。输出 2x2 块写回时按
 *   OH/OW 裁剪; 输入块起点为 P[2·ty], P[2·tx] (P 为 1 像素 0 填充缓冲)。
 *
 * 实现以通用 -O3 编译, 变换为标量/加法, 跨通道批量累加由编译器向量化。 */
#include "acfun.h"
#include "internal.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

/* 单通道 4x4 输入块 -> 16 个 Winograd 域值 (d_hat = BT d BTᵀ, 可分离)。
 * d[16] 为 row-major 4x4; dh[16] 为列主序 (dh[i*4+j]) 以匹配 g_hat 布局。 */
inline void input_transform(const float* d, float* dh) {
    /* U[i][l] = (BT d)[i][l], B 行:
     * i0:[1,0,-1,0]  i1:[0,1,1,0]  i2:[0,-1,1,0]  i3:[0,-1,0,1] */
    float U[4][4];
    for (int l = 0; l < 4; ++l) {
        float c0 = d[l], c1 = d[4 + l], c2 = d[8 + l], c3 = d[12 + l];
        U[0][l] =  c0 - c2;
        U[1][l] =  c1 + c2;
        U[2][l] = -c1 + c2;
        U[3][l] = -c1 + c3;
    }
    /* d_hat[i][j] = Σ_l U[i][l] · BT[j][l]; 各列 j 的系数 (BT[j][l]):
     * j0:[1,0,-1,0]  j1:[0,1,1,0]  j2:[0,-1,1,0]  j3:[0,-1,0,1] */
    for (int i = 0; i < 4; ++i) {
        float u0 = U[i][0], u1 = U[i][1], u2 = U[i][2], u3 = U[i][3];
        dh[i * 4 + 0] =  u0 - u2;
        dh[i * 4 + 1] =  u1 + u2;
        dh[i * 4 + 2] = -u1 + u2;
        dh[i * 4 + 3] = -u1 + u3;
    }
}

/* 单通道 3x3 权重 -> 16 个 Winograd 域值 (g_hat = G g Gᵀ, 可分离)。
 * g[9] 为 row-major 3x3; gh[16] 为列主序 (gh[i*4+j])。
 * 注意 G 为 4x3, 故 g_hat[i][j] = Σ_{y=0..2} T[i][y]·G[j][y], 仅含 t0/t1/t2,
 * 绝不可读取 T[i][3] (越界, 会取栈上垃圾值导致结果发散)。 */
inline void filter_transform(const float* g, float* gh) {
    /* T[i][y] = Σ_x G[i][x] g[x][y],  G 行:
     * i0:[1,0,0]  i1:[.5,.5,.5]  i2:[.5,-.5,.5]  i3:[0,0,1] */
    float T[4][3];
    for (int y = 0; y < 3; ++y) {
        float a = g[y], b = g[3 + y], c = g[6 + y];
        T[0][y] = a;
        T[1][y] = 0.5f * (a + b + c);
        T[2][y] = 0.5f * (a - b + c);
        T[3][y] = c;
    }
    /* g_hat[i][j] = Σ_y T[i][y] G[j][y];  G 列 j 的系数 (G 仅 3 列):
     * j0:[1, 0, 0]   j1:[.5,.5,.5]   j2:[.5,-.5,.5]   j3:[0, 0, 1] */
    for (int i = 0; i < 4; ++i) {
        float t0 = T[i][0], t1 = T[i][1], t2 = T[i][2];
        gh[i * 4 + 0] =  t0;                       /* j0: [1,0,0] */
        gh[i * 4 + 1] =  0.5f * (t0 + t1 + t2);    /* j1: [.5,.5,.5] */
        gh[i * 4 + 2] =  0.5f * (t0 - t1 + t2);    /* j2: [.5,-.5,.5] */
        gh[i * 4 + 3] =  t2;                       /* j3: [0,0,1] */
    }
}

/* 输出变换 Y = AT M ATᵀ (M = m_hat[c] 为 4x4, 以 ij*Cout+c 跨 c 步长存储), 写回 2x2。
 * AT(2x4):
 *   [1 1 1 0]
 *   [0 1 -1 1] */
inline void output_transform(const float* mh, int Cout, int c, float Y[4]) {
    /* Y[r*2+c] = Σ_{i,j} AT[r][i] · mh[(i*4+j)*Cout + c] · AT[c][j] */
    static const float AT[2][4] = {{1.f, 1.f, 1.f, 0.f}, {0.f, 1.f, -1.f, 1.f}};
    for (int r = 0; r < 2; ++r) {
        for (int cc = 0; cc < 2; ++cc) {
            float s = 0.f;
            for (int i = 0; i < 4; ++i) {
                float ri = AT[r][i];
                if (ri == 0.f) continue;
                for (int j = 0; j < 4; ++j) {
                    float cj = AT[cc][j];
                    if (cj == 0.f) continue;
                    s += ri * cj * mh[((size_t)i * 4 + j) * Cout + c];
                }
            }
            Y[r * 2 + cc] = s;
        }
    }
}

} /* namespace */

/* Winograd F(2,3) 卷积主体: 3x3 / stride1 / pad1, NHWC。
 * 要求 H>=3 && W>=3; act=1 融合 ReLU。权重布局: w[KH*KW*Cin*Cout]。 */
void acfun_winograd_conv3x3(const float* input, int N, int H, int W, int Cin,
                            const float* weight, const float* bias,
                            int Cout, float* output, int act) {
    const int OH = H, OW = W;                 /* s1 p1 => 输出同尺寸 */
    const int tH = (OH + 1) / 2;              /* 2x2 输出瓦片数 (向上取整) */
    const int tW = (OW + 1) / 2;

    /* 预计算变换后权重 g_hat[16][Cin][Cout], 内层 c 连续便于向量化累加。
     * 布局: g_hat[((ij*Cin + d) * Cout) + c]。 */
    std::vector<float> g_hat((size_t)16 * Cin * Cout);
    for (int d = 0; d < Cin; ++d) {
        for (int c = 0; c < Cout; ++c) {
            float g[9];
            for (int k = 0; k < 9; ++k)
                g[k] = weight[((size_t)k * Cin + d) * Cout + c];
            float gh[16];
            filter_transform(g, gh);
            for (int ij = 0; ij < 16; ++ij)
                g_hat[((size_t)ij * Cin + d) * Cout + c] = gh[ij];
        }
    }

    std::vector<float> m_hat((size_t)16 * Cout);   /* 当前瓦片的输出域累加 */

    /* 带 1 像素 0 填充的输入缓冲 (尺寸 (H+2)x(W+2)xCin), 边界瓦片越界读取
     * 自然为 0, 简化裁剪逻辑并保证正确性。每个 batch 项按需重建。 */
    std::vector<float> pad((size_t)(H + 2) * (W + 2) * Cin);
    const int pstride = (W + 2) * Cin;

    for (int n = 0; n < N; ++n) {
        const float* x = input + (size_t)n * H * W * Cin;
        /* 重建该 batch 项的填充缓冲 (以 x 而非 input 为源, 保证 n>0 正确) */
        std::memset(pad.data(), 0, pad.size() * sizeof(float));
        for (int ih = 0; ih < H; ++ih)
            for (int iw = 0; iw < W; ++iw)
                memcpy(&pad[((size_t)(ih + 1) * (W + 2) + (iw + 1)) * Cin],
                       x + ((size_t)ih * W + iw) * Cin, Cin * sizeof(float));
        const float* pbase = pad.data();
        /* 逐瓦片处理; 边界瓦片越界读取经 pad 缓冲得到 0, 输出写回时按 OH/OW 裁剪 */
        for (int ty = 0; ty < tH; ++ty) {
            int oh0 = 2 * ty;                 /* 瓦片输出行起点 (pad 缓冲行 2*ty..2*ty+3) */
            for (int tx = 0; tx < tW; ++tx) {
                int ow0 = 2 * tx;
                std::memset(m_hat.data(), 0, (size_t)16 * Cout * sizeof(float));

                /* 沿输入通道累加进输出域 */
                for (int d = 0; d < Cin; ++d) {
                    /* 提取 4x4 输入块 (pad 缓冲), 起始 pad 行 = 2*ty, 起始 pad 列 = 2*tx */
                    int pr0 = 2 * ty, pc0 = 2 * tx;
                    float dblock[16];
                    for (int k = 0; k < 4; ++k) {
                        int pr = pr0 + k;
                        const float* prow = (pr >= 0 && pr < H + 2)
                            ? &pbase[(size_t)pr * pstride] : nullptr;
                        for (int l = 0; l < 4; ++l) {
                            int pc = pc0 + l;
                            if (prow && pc >= 0 && pc < W + 2)
                                dblock[k * 4 + l] = prow[(size_t)pc * Cin + d];
                            else
                                dblock[k * 4 + l] = 0.0f;
                        }
                    }
                    float dh[16];
                    input_transform(dblock, dh);

                    /* m_hat[ij][c] += dh[ij] * g_hat[ij][d][c] (c 连续) */
                    for (int ij = 0; ij < 16; ++ij) {
                        const float* gh = &g_hat[((size_t)ij * Cin + d) * Cout];
                        float*       mh = &m_hat[(size_t)ij * Cout];
                        float        v  = dh[ij];
                        for (int c = 0; c < Cout; ++c) mh[c] += v * gh[c];
                    }
                }

                /* 输出变换 Y = AT M ATᵀ (M = m_hat[c]), 写回 2x2 输出并融合 bias/ReLU */
                float* yo = output + (size_t)n * OH * OW * Cout;
                for (int c = 0; c < Cout; ++c) {
                    float Y[4];
                    output_transform(m_hat.data(), Cout, c, Y);  /* M 列主序: m[i*4+j] = m_hat[i*Cout+j] */
                    float b = bias ? bias[c] : 0.0f;
                    for (int r = 0; r < 2; ++r) {
                        int oh = oh0 + r;
                        if (oh >= OH) continue;
                        for (int cc = 0; cc < 2; ++cc) {
                            int ow = ow0 + cc;
                            if (ow >= OW) continue;
                            float v = Y[r * 2 + cc] + b;
                            if (act == 1 && v < 0.0f) v = 0.0f;
                            yo[(size_t)(oh * OW + ow) * Cout + c] = v;
                        }
                    }
                }
            }
        }
    }
}
