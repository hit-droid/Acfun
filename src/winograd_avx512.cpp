/* Acfun Winograd 卷积 AVX-512+FMA 变体 (F(2,3), 3x3 / stride1 / pad1)。
 *
 * 与 winograd.cpp 标量实现、winograd_avx2.cpp 同算法, 仅在最热的「量化/累加内层」
 *   m_hat[ij][c] += dh[ij] * g_hat[ij][d][c]
 * 上做三层优化, 这正是 MNN 在 Conv 上压低延迟的关键:
 *
 *   1) 宽度向量化: 512-bit FMA, 一次 16 通道 (c 连续), 较 AVX2 的 8 通道翻倍吞吐;
 *   2) 寄存器驻留累加: 以 "c 维 16 路分块" 为最外层, 每个 c 块内用 16 个 zmm 累加器
 *      (acc[ij]) 跨全部输入通道 d 累加、仅末了写回 m_hat 一次, 彻底消除旧标量/AVX2
 *      路径 "每 (d,ij) 重读/重写整条 m_hat[c]" 的冗余访存。dh[d][16] 每通道只算一次
 *      并缓存于 dh_all, 瓦片间复用预分配的 m_hat/dh_all/pad 缓冲。
 *   3) 瓦片并行: 输出 2x2 瓦片彼此独立 (写入互不相交的输出区), 故对 (ty,tx) 瓦片网格做
 *      OpenMP 并行; 每线程持私有 m_hat/dh_all, g_hat(仅依赖权重, 调用内只读) 与 pad
 *      (逐图串行填充后只读) 跨线程共享。受 cgroup CPU 配额约束, 大卷积最多 ~核心数倍
 *      加速。小卷积 (工作量低于阈值) 回落纯串行, 避免 fork/join 开销拖累微型模型。
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

/* 单瓦片计算 (2x2 输出块): 使用调用方提供的线程私有 m_hat(16*Cout) 与 dh_all(Cin*16)。
 * g_hat(仅依赖权重, 调用内只读) 与 pad(逐图串行填充后只读) 跨线程共享。 */
static void wino_tile_avx512(int ty, int tx, int OH, int OW, int Cin, int Cout, int act,
                             const float* bias,
                             const float* g_hat, const float* pbase, int pstride,
                             float* m_hat, float* dh_all, float* output, int n) {
    int oh0 = 2 * ty, ow0 = 2 * tx;

    /* 先对每个通道 d 算好输入变换 dh[d][16] (一次提取+变换, 瓦片内复用) */
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
        input_transform(dblock, &dh_all[(size_t)d * 16]);
    }

    /* c 维 16 路分块; 每个 c 块内以 16 个 zmm 累加器 (acc[ij]) 跨全部 d 累加,
     * 末了写回 m_hat 一次。m_hat 被完整覆盖, 无需 memset。 */
    int c = 0;
    for (; c + 16 <= Cout; c += 16) {
        __m512 acc[16];
        for (int ij = 0; ij < 16; ++ij) acc[ij] = _mm512_setzero_ps();
        for (int d = 0; d < Cin; ++d) {
            const float* dh = &dh_all[(size_t)d * 16];
            for (int ij = 0; ij < 16; ++ij) {
                __m512 g = _mm512_loadu_ps(&g_hat[((size_t)ij * Cin + d) * Cout + c]);
                acc[ij] = _mm512_fmadd_ps(_mm512_set1_ps(dh[ij]), g, acc[ij]);
            }
        }
        for (int ij = 0; ij < 16; ++ij)
            _mm512_storeu_ps(&m_hat[(size_t)ij * Cout + c], acc[ij]);
    }
    if (c < Cout) {                               /* 余数列块: masking 收尾 */
        int rem = Cout - c;
        __mmask16 msk = (__mmask16)((1u << rem) - 1);
        __m512 acc[16];
        for (int ij = 0; ij < 16; ++ij) acc[ij] = _mm512_setzero_ps();
        for (int d = 0; d < Cin; ++d) {
            const float* dh = &dh_all[(size_t)d * 16];
            for (int ij = 0; ij < 16; ++ij) {
                __m512 g = _mm512_maskz_loadu_ps(msk, &g_hat[((size_t)ij * Cin + d) * Cout + c]);
                acc[ij] = _mm512_fmadd_ps(_mm512_set1_ps(dh[ij]), g, acc[ij]);
            }
        }
        for (int ij = 0; ij < 16; ++ij)
            _mm512_mask_storeu_ps(&m_hat[(size_t)ij * Cout + c], msk, acc[ij]);
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

void acfun_winograd_conv3x3_avx512(const float* input, int N, int H, int W, int Cin,
                             const float* weight, const float* bias,
                             int Cout, float* output, int act) {
    const int OH = H, OW = W;                 /* s1 p1 => 输出同尺寸 */
    const int tH = (OH + 1) / 2;              /* 2x2 输出瓦片数 (向上取整) */
    const int tW = (OW + 1) / 2;

    /* 预计算变换后权重 g_hat[16][Cin][Cout], 内层 c 连续。布局: ((ij*Cin+d)*Cout)+c。
     * 仅依赖权重, 调用内计算一次, 跨所有瓦片/线程只读共享。 */
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

    /* 带 1 像素 0 填充的输入缓冲 (尺寸 (H+2)x(W+2)xCin), 逐图填充后只读共享。 */
    std::vector<float> pad((size_t)(H + 2) * (W + 2) * Cin);
    const int pstride = (W + 2) * Cin;

    /* 仅大卷积并行: 避免微型模型 (e2e 32x32 等) 的 OpenMP fork/join 开销拖累延迟。 */
    const long long work = (long long)Cin * Cout * OH * OW;
    const int PARALLEL = (work >= 8000000LL);
    int nth = acfun_effective_cores();          /* cgroup 感知: 受限容器下压到配额内 */
    if (nth > tH * tW) nth = tH * tW;           /* 不过度订阅瓦片 */
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
                std::vector<float> dh_all((size_t)Cin * 16);
                #pragma omp for collapse(2) schedule(static)
                for (int ty = 0; ty < tH; ++ty)
                    for (int tx = 0; tx < tW; ++tx)
                        wino_tile_avx512(ty, tx, OH, OW, Cin, Cout, act, bias,
                                         g_hat.data(), pbase, pstride,
                                         m_hat.data(), dh_all.data(), output, n);
            }
        } else {
            std::vector<float> m_hat((size_t)16 * Cout);
            std::vector<float> dh_all((size_t)Cin * 16);
            for (int ty = 0; ty < tH; ++ty)
                for (int tx = 0; tx < tW; ++tx)
                    wino_tile_avx512(ty, tx, OH, OW, Cin, Cout, act, bias,
                                     g_hat.data(), pbase, pstride,
                                     m_hat.data(), dh_all.data(), output, n);
        }
    }
}
