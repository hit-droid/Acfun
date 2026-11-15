/* Acfun 最大池化 — AVX2 (256-bit) 通道向量化后端。
 *
 * 与 AVX-512 后端同构, 仅把通道向量宽度降为 8 路 (__m256 / vmaxps), 供不支持 AVX-512 的
 * 主机 (仅 AVX2+FMA) 使用; 运行时由 layers.cpp 按能力在 AVX-512 / AVX2 / 标量间派发。
 * 与标量版严格等价 (越界位置跳过、post_relu 就地 ReLU)。编译单元必须带 -mavx2 -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <cstddef>

void acfun_maxpool_avx2(const float* in, int N, int H, int W, int C,
                        int pool, int stride, int pad, float* out, int post_relu) {
    int OH = (H + 2 * pad - pool) / stride + 1;
    int OW = (W + 2 * pad - pool) / stride + 1;
    if (OH <= 0 || OW <= 0) return;

    const float sentinel = -1e30f;
    __m256 vsent = _mm256_set1_ps(sentinel);
    __m256 vzero = _mm256_setzero_ps();

    for (int n = 0; n < N; ++n) {
        const float* inb = in + (size_t)n * H * W * C;
        float* outb = out + (size_t)n * OH * OW * C;
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                int cFull = C & ~(8 - 1);   /* 8 路对齐的通道数 */
                int c;
                for (c = 0; c < cFull; c += 8) {
                    __m256 acc = vsent;
                    for (int ph = 0; ph < pool; ++ph) {
                        int ih = oh * stride - pad + ph;
                        if (ih < 0 || ih >= H) continue;
                        for (int pw = 0; pw < pool; ++pw) {
                            int iw = ow * stride - pad + pw;
                            if (iw < 0 || iw >= W) continue;
                            const float* p = inb + ((size_t)ih * W + iw) * C + c;
                            acc = _mm256_max_ps(acc, _mm256_loadu_ps(p));
                        }
                    }
                    if (post_relu) acc = _mm256_max_ps(acc, vzero);
                    _mm256_storeu_ps(outb + ((size_t)(oh * OW + ow) * C + c), acc);
                }
                /* 尾部不足 8 通道: 标量归约 (与标量版算法一致) */
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
