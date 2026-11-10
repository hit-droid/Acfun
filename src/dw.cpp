/* Acfun 深度(逐通道)卷积 — 调度器 + 标量参考。
 * 通用 -O3 编译; AVX-512 内核在单独编译单元中实现, 运行时按 CPU 特性派发。 */
#include "acfun.h"
#include "internal.h"
#include <cmath>
#include <cstring>

/* 标量参考 (正确性基线 + 可移植兜底): 权重布局 [C, KH, KW]。 */
void acfun_conv2d_dw_scalar(const float* input, int N, int H, int W, int C,
                            const float* weight, const float* bias,
                            int KH, int KW, int stride, int pad,
                            float* output, int act) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    for (int n = 0; n < N; ++n) {
        const float* in  = input  + (size_t)n * H * W * C;
        float*       out = output + (size_t)n * OH * OW * C;
        for (int oc = 0; oc < C; ++oc) {
            const float* wch = weight + (size_t)oc * KH * KW;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    float s = bias ? bias[oc] : 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        int ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = ow * stride - pad + kw;
                            if (iw < 0 || iw >= W) continue;
                            s += in[((size_t)ih * W + iw) * C + oc] * wch[kh * KW + kw];
                        }
                    }
                    if (act == 1 && s < 0.0f) s = 0.0f;
                    out[((size_t)oh * OW + ow) * C + oc] = s;
                }
            }
        }
    }
}

/* 调度器: 检测到 avx512f 走 512-bit 向量化, 否则标量。 */
void acfun_conv2d_dw(const float* input, int N, int H, int W, int C,
                     const float* weight, const float* bias,
                     int KH, int KW, int stride, int pad,
                     float* output, int act) {
    if (KH <= 0 || KW <= 0) return;
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx512f")) {
        acfun_conv2d_dw_avx512(input, N, H, W, C, weight, bias, KH, KW, stride, pad, output, act);
        return;
    }
#endif
    acfun_conv2d_dw_scalar(input, N, H, W, C, weight, bias, KH, KW, stride, pad, output, act);
}
