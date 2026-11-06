/* Acfun Winograd 卷积 AVX2+FMA 变体 (F(2,3), 3x3 / stride1 / pad1)。
 *
 * 与 winograd.cpp 标量实现同算法, 仅把最热的「量化/累加内层」
 *   m_hat[ij][c] += dh[ij] * g_hat[ij][d][c]
 * 用 256-bit AVX2 + FMA 向量化 (c 连续, 每次 8 通道)。该循环占单瓦片算力主体
 * (16·Cin·Cout MAC/瓦片, 远超输入/输出变换的 16·Cin / 16·Cout 加法), 是 Winograd
 * 路径的首要热点。
 *
 * 性能优化沿用与 AVX-512 变体一致的两层: (a) 无分支 ReLU (vmax 语义) 融合, 消除
 * 数据相关分支惩罚; (b) 输出 2x2 瓦片并行 — 瓦片彼此独立, 对 (ty,tx) 网格做 OpenMP
 * 并行, 每线程持私有 m_hat, g_hat(仅依赖权重, 调用内只读) 与 pad(逐图串行填充后只读)
 * 跨线程共享; 仅大卷积 (工作量超阈值) 并行, 微型模型回落串行避免 fork/join 开销。
 *
 * 编译单元需带 -mavx2 -mfma (见 CMakeLists.txt); 仅在本函数内使用 AVX2 指令。 */
#include "acfun.h"
#include "internal.h"
#include <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

namespace {

/* 单通道 4x4 输入块 -> 16 个 Winograd 域值 (d_hat = BT d BTᵀ, 可分离)。 */
inline void input_transform(const float* d, float* dh) {
    float U[4][4];
    for (int l = 0; l < 4; ++l) {
        float c0 = d[l], c1 = d[4 + l], c2 = d[8 + l], c3 = d[12 + l];
        U[0][l] =  c0 - c2;
        U[1][l] =  c1 + c2;
        U[2][l] = -c1 + c2;
        U[3][l] = -c1 + c3;
    }
    for (int i = 0; i < 4; ++i) {
        float u0 = U[i][0], u1 = U[i][1], u2 = U[i][2], u3 = U[i][3];
        dh[i * 4 + 0] =  u0 - u2;
        dh[i * 4 + 1] =  u1 + u2;
        dh[i * 4 + 2] = -u1 + u2;
        dh[i * 4 + 3] = -u1 + u3;
    }
}

/* 单通道 3x3 权重 -> 16 个 Winograd 域值 (g_hat = G g Gᵀ, 可分离)。 */
inline void filter_transform(const float* g, float* gh) {
    float T[4][3];
    for (int y = 0; y < 3; ++y) {
        float a = g[y], b = g[3 + y], c = g[6 + y];
        T[0][y] = a;
        T[1][y] = 0.5f * (a + b + c);
        T[2][y] = 0.5f * (a - b + c);
        T[3][y] = c;
    }
    for (int i = 0; i < 4; ++i) {
        float t0 = T[i][0], t1 = T[i][1], t2 = T[i][2];
        gh[i * 4 + 0] =  t0;
        gh[i * 4 + 1] =  0.5f * (t0 + t1 + t2);
        gh[i * 4 + 2] =  0.5f * (t0 - t1 + t2);
        gh[i * 4 + 3] =  t2;
    }
}

/* 输出变换 Y = AT M ATᵀ (M = m_hat[c] 为 4x4, 以 ij*Cout+c 跨 c 步长存储)。 */
inline void output_transform(const float* mh, int Cout, int c, float Y[4]) {
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

/* 单瓦片计算 (2x2 输出块): 使用调用方提供的线程私有 m_hat(16*Cout)。
 * g_hat(仅依赖权重, 调用内只读) 与 pad(逐图串行填充后只读) 跨线程共享。 */
static void wino_tile_avx2(int ty, int tx, int OH, int OW, int Cin, int Cout, int act,
                           const float* bias,
                           const float* g_hat, const float* pbase, int pstride,
                           float* m_hat, float* output, int n) {
    int oh0 = 2 * ty, ow0 = 2 * tx;
    std::memset(m_hat, 0, (size_t)16 * Cout * sizeof(float));

    for (int d = 0; d < Cin; ++d) {
        int pr0 = 2 * ty, pc0 = 2 * tx;
        float dblock[16];
        for (int k = 0; k < 4; ++k) {
            int pr = pr0 + k;
            const float* prow = (pr >= 0 && pr < OH + 2)
                ? &pbase[(size_t)pr * pstride] : nullptr;
            for (int l = 0; l < 4; ++l) {
                int pc = pc0 + l;
                if (prow && pc >= 0 && pc < OW + 2)
                    dblock[k * 4 + l] = prow[(size_t)pc * Cin + d];
                else
                    dblock[k * 4 + l] = 0.0f;
            }
        }
        float dh[16];
        input_transform(dblock, dh);

        /* 量化/累加内层: 256-bit AVX2 向量化 (c 连续, 每次 8 通道) */
        for (int ij = 0; ij < 16; ++ij) {
            const float* gh = &g_hat[((size_t)ij * Cin + d) * Cout];
            float*       mh = &m_hat[(size_t)ij * Cout];
            float        v  = dh[ij];
            __m256 vv = _mm256_set1_ps(v);
            int c = 0;
            for (; c + 8 <= Cout; c += 8) {
                __m256 gv = _mm256_loadu_ps(gh + c);
                __m256 mv = _mm256_loadu_ps(mh + c);
                _mm256_storeu_ps(mh + c, _mm256_fmadd_ps(vv, gv, mv));
            }
            for (; c < Cout; ++c) mh[c] += v * gh[c];
        }
    }

    /* 输出变换 Y = AT M ATᵀ, 写回 2x2 输出并融合 bias/ReLU (无分支 vmax) */
    float* yo = output + (size_t)n * OH * OW * Cout;
    for (int c = 0; c < Cout; ++c) {
        float Y[4];
        output_transform(m_hat, Cout, c, Y);
        float b = bias ? bias[c] : 0.0f;
        for (int r = 0; r < 2; ++r) {
            int oh = oh0 + r;
            if (oh >= OH) continue;
            for (int cc = 0; cc < 2; ++cc) {
                int ow = ow0 + cc;
                if (ow >= OW) continue;
                float v = Y[r * 2 + cc] + b;
                if (act == 1) v = v < 0.0f ? 0.0f : v;
                yo[(size_t)(oh * OW + ow) * Cout + c] = v;
            }
        }
    }
}

void acfun_winograd_conv3x3_avx2(const float* input, int N, int H, int W, int Cin,
                                 const float* weight, const float* bias,
                                 int Cout, float* output, int act) {
    const int OH = H, OW = W;
    const int tH = (OH + 1) / 2;
    const int tW = (OW + 1) / 2;

    /* 预计算变换后权重 g_hat[16][Cin][Cout] (内层 c 连续), 调用内只读共享。 */
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
                std::vector<float> m_hat((size_t)16 * Cout);
                #pragma omp for collapse(2) schedule(static)
                for (int ty = 0; ty < tH; ++ty)
                    for (int tx = 0; tx < tW; ++tx)
                        wino_tile_avx2(ty, tx, OH, OW, Cin, Cout, act, bias,
                                       g_hat.data(), pbase, pstride,
                                       m_hat.data(), output, n);
            }
        } else {
            std::vector<float> m_hat((size_t)16 * Cout);
            for (int ty = 0; ty < tH; ++ty)
                for (int tx = 0; tx < tW; ++tx)
                    wino_tile_avx2(ty, tx, OH, OW, Cin, Cout, act, bias,
                                   g_hat.data(), pbase, pstride,
                                   m_hat.data(), output, n);
        }
    }
}
