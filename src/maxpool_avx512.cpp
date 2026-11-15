/* Acfun 最大池化 — AVX-512 (512-bit) 通道向量化后端。
 *
 * 池化的内层维度是通道 C: 对每个输出像素 (oh,ow), 在其 pool×pool 空间窗口内对 C 个通道
 * 各取最大值。原实现 (layers.cpp 标量版) 是「输出像素 × 通道」的 6 层嵌套标量循环, 通道维度
 * 完全没有向量化。本后端把通道打包成 16 路 zmm, 一次 vmaxps 同时归约 16 个通道的窗口最大值,
 * 较标量 (每通道一次标量比较) 在宽通道 (C≫16) 下逼近 16× 的通道归约提速。
 *
 * 与标量版严格等价: 越界窗口位置 (pad 引入) 直接跳过 (不载入, 累加器保持 -1e30 哨兵),
 * 与标量版「越界 continue」完全一致; post_relu 在池化后就地 ReLU (max(x,0)), 与先 Pool 再
 * 独立 ReLU 数学等价。编译单元必须带 -mavx512f (运行时由 layers.cpp 按 avx512f 能力派发)。 */
#include "internal.h"
#include <immintrin.h>
#include <cstddef>

void acfun_maxpool_avx512(const float* in, int N, int H, int W, int C,
                          int pool, int stride, int pad, float* out, int post_relu) {
    int OH = (H + 2 * pad - pool) / stride + 1;
    int OW = (W + 2 * pad - pool) / stride + 1;
    if (OH <= 0 || OW <= 0) return;

    const float sentinel = -1e30f;
    __m512 vsent = _mm512_set1_ps(sentinel);
    __m512 vzero = _mm512_setzero_ps();

    for (int n = 0; n < N; ++n) {
        const float* inb = in + (size_t)n * H * W * C;
        float* outb = out + (size_t)n * OH * OW * C;
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int cFull = C & ~(16 - 1);   /* 16 路对齐的通道数 */
                int c;
                for (c = 0; c < cFull; c += 16) {
                    __m512 acc = vsent;
                    for (int ph = 0; ph < pool; ++ph) {
                        int ih = oh * stride - pad + ph;
                        if (ih < 0 || ih >= H) continue;
                        for (int pw = 0; pw < pool; ++pw) {
                            int iw = ow * stride - pad + pw;
                            if (iw < 0 || iw >= W) continue;
                            const float* p = inb + ((size_t)ih * W + iw) * C + c;
                            acc = _mm512_max_ps(acc, _mm512_loadu_ps(p));
                        }
                    }
                    if (post_relu) acc = _mm512_max_ps(acc, vzero);
                    _mm512_storeu_ps(outb + ((size_t)(oh * OW + ow) * C + c), acc);
                }
                /* 尾部不足 16 通道: 标量归约 (与标量版算法一致) */
                for (; c < C; ++c) {
                    float mx = sentinel;
                    for (int ph = 0; ph < pool; ++ph) {
                        int ih = oh * stride - pad + ph;
                        if (ih < 0 || ih >= H) continue;
                        for (int pw = 0; pw < pool; ++pw) {
                            int iw = ow * stride - pad + pw;
                            if (iw < 0 || iw >= W) continue;
                            float v = inb[((size_t)ih * W + iw) * C + c];
                            if (v > mx) mx = v;
                        }
                    }
                    if (post_relu && mx < 0.0f) mx = 0.0f;
                    outb[(size_t)(oh * OW + ow) * C + c] = mx;
                }
            }
        }
    }
}
