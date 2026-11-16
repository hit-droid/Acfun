/* Acfun 分组卷积 — AVX2+FMA 直接卷积内核。
 *
 * 与 AVX-512 版同构, 输出通道 oc 打包 8 路 ymm。尾部用标量处理 (Cout_p 非 8 倍数),
 * 保证任意通道数下与标量参考逐元素等价。编译单元带 -mavx2 -mfma;
 * 由 group_conv.cpp 在 8<=Cout_per<16 或仅支持 avx2 时派发。 */
#include "internal.h"
#include <immintrin.h>
#include <cstddef>

void acfun_group_conv_avx2(const float* input, int N, int H, int W, int Cin,
                           const float* weight, const float* bias,
                           int KH, int KW, int Cout, int stride, int pad,
                           int groups, float* output, int act) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    const int VL = 8;
    int Cin_p = Cin / groups, Cout_p = Cout / groups;
    int ocFull = Cout_p & ~(VL - 1);
    const size_t wg_stride = (size_t)Cout_p * KH * KW * Cin_p;
    __m256 vzero = _mm256_setzero_ps();

    for (int n = 0; n < N; ++n) {
        const float* inb = input + (size_t)n * H * W * Cin;
        float* outb = output + (size_t)n * OH * OW * Cout;
        for (int g = 0; g < groups; ++g) {
            const float* wg = weight + (size_t)g * wg_stride;
            const float* bg = bias ? bias + (size_t)g * Cout_p : nullptr;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    size_t opix = ((size_t)oh * OW + ow) * Cout + (size_t)g * Cout_p;
                    int oc = 0;
                    for (; oc < ocFull; oc += VL) {
                        __m256 acc = bg ? _mm256_loadu_ps(bg + oc) : vzero;
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * stride - pad + kw;
                                if (iw < 0 || iw >= W) continue;
                                const float* inpx = inb + ((size_t)ih * W + iw) * Cin + (size_t)g * Cin_p;
                                for (int ic = 0; ic < Cin_p; ++ic) {
                                    __m256 a = _mm256_set1_ps(inpx[ic]);
                                    const float* wp = wg + (size_t)((kh * KW + kw) * Cin_p + ic) * Cout_p + oc;
                                    acc = _mm256_fmadd_ps(a, _mm256_loadu_ps(wp), acc);
                                }
                            }
                        }
                        if (act == 1) acc = _mm256_max_ps(acc, vzero);
                        _mm256_storeu_ps(outb + opix + oc, acc);
                    }
                    /* 尾部不足 8 通道: 标量归约, 与标量参考算法一致 */
                    for (; oc < Cout_p; ++oc) {
                        float s = bg ? bg[oc] : 0.0f;
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * stride - pad + kw;
                                if (iw < 0 || iw >= W) continue;
                                const float* inpx = inb + ((size_t)ih * W + iw) * Cin + (size_t)g * Cin_p;
                                const float* wp = wg + (size_t)((kh * KW + kw) * Cin_p) * Cout_p + oc;
                                for (int ic = 0; ic < Cin_p; ++ic)
                                    s += inpx[ic] * wp[(size_t)ic * Cout_p];
                            }
                        }
                        if (act == 1 && s < 0.0f) s = 0.0f;
                        outb[opix + oc] = s;
                    }
                }
            }
        }
    }
}
