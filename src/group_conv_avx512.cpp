/* Acfun 分组卷积 — AVX-512F+FMA 直接卷积内核。
 *
 * 以每组输出通道 Cout_per 为向量维度 (权重 w[..*Cout_per + oc] 与输出 NHWC 的 oc 均连续):
 * 每个输出像素对 oc 打包 16 路 zmm, 外层 kh->kw->ic 逐 tap 广播单个输入标量, 与 16 路权重
 * 做 FMA 累加。相较旧「逐组抽取 + im2col + 小 GEMM + 拷回」路径, 无中间缓冲、无 gather,
 * 且直接命中 NHWC 访存; 累加顺序 (kh,kw,ic) 与标量参考一致, 数值等价。
 * 编译单元需带 -mavx512f -mfma; 由 group_conv.cpp 在 Cout_per>=16 且支持 avx512f 时派发。 */
#include "internal.h"
#include <immintrin.h>
#include <cstddef>

void acfun_group_conv_avx512(const float* input, int N, int H, int W, int Cin,
                             const float* weight, const float* bias,
                             int KH, int KW, int Cout, int stride, int pad,
                             int groups, float* output, int act) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    const int VL = 16;
    int Cin_p = Cin / groups, Cout_p = Cout / groups;
    int ocFull = Cout_p & ~(VL - 1);
    const size_t wg_stride = (size_t)Cout_p * KH * KW * Cin_p;
    __m512 vzero = _mm512_setzero_ps();

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
                        __m512 acc = bg ? _mm512_loadu_ps(bg + oc) : vzero;
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * stride - pad + kw;
                                if (iw < 0 || iw >= W) continue;
                                const float* inpx = inb + ((size_t)ih * W + iw) * Cin + (size_t)g * Cin_p;
                                for (int ic = 0; ic < Cin_p; ++ic) {
                                    __m512 a = _mm512_set1_ps(inpx[ic]);
                                    const float* wp = wg + (size_t)((kh * KW + kw) * Cin_p + ic) * Cout_p + oc;
                                    acc = _mm512_fmadd_ps(a, _mm512_loadu_ps(wp), acc);
                                }
                            }
                        }
                        if (act == 1) acc = _mm512_max_ps(acc, vzero);
                        _mm512_storeu_ps(outb + opix + oc, acc);
                    }
                    /* 尾部不足 16 通道: 掩码向量处理, 与标量等价 */
                    if (oc < Cout_p) {
                        __mmask16 m = (__mmask16)((1u << (Cout_p - oc)) - 1);
                        __m512 acc = bg ? _mm512_maskz_loadu_ps(m, bg + oc) : vzero;
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * stride - pad + kw;
                                if (iw < 0 || iw >= W) continue;
                                const float* inpx = inb + ((size_t)ih * W + iw) * Cin + (size_t)g * Cin_p;
                                for (int ic = 0; ic < Cin_p; ++ic) {
                                    __m512 a = _mm512_set1_ps(inpx[ic]);
                                    const float* wp = wg + (size_t)((kh * KW + kw) * Cin_p + ic) * Cout_p + oc;
                                    acc = _mm512_maskz_fmadd_ps(m, a, _mm512_maskz_loadu_ps(m, wp), acc);
                                }
                            }
                        }
                        if (act == 1) acc = _mm512_max_ps(acc, vzero);
                        _mm512_mask_storeu_ps(outb + opix + oc, m, acc);
                    }
                }
            }
        }
    }
}
