/* Acfun 分组卷积 — 直接卷积调度器 + 标量参考。
 *
 * 一般分组卷积 (groups>1, 非深度) 旧实现在 layers.cpp 中逐组把 NHWC 输入抽取到连续缓冲、
 * 走 im2col+GEMM、再把输出拷回。当每组通道数很小时 (分组卷积的典型情形, 如 Cin_per=4),
 * 这套「抽取 + im2col + 小 GEMM + 拷回」的固定开销远超卷积本身的算力, 甚至比朴素直乘还慢
 * (实测 14x14x24->48 g6 仅 0.2x)。本路径改为**直接卷积**: 原地读 NHWC 输入、按输出通道
 * 向量化、写回 NHWC 输出, 完全省去中间缓冲与 im2col。
 *
 * 权重布局 (与既有引擎/朴素参考一致): [groups][KH*KW*Cin_per][Cout_per], 即
 *   weight[(g*KH*KW*Cin_per + (kh*KW+kw)*Cin_per + ic)*Cout_per + oc], oc 连续。
 * 累加顺序为 kh -> kw -> ic, 与朴素参考一致, 故 SIMD 与标量结果逐元素几乎相同。 */
#include "acfun.h"
#include "internal.h"
#include <cstddef>

void acfun_group_conv_scalar(const float* input, int N, int H, int W, int Cin,
                             const float* weight, const float* bias,
                             int KH, int KW, int Cout, int stride, int pad,
                             int groups, float* output, int act) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    int Cin_p = Cin / groups, Cout_p = Cout / groups;
    const size_t wg_stride = (size_t)Cout_p * KH * KW * Cin_p;
    for (int n = 0; n < N; ++n) {
        const float* inb = input + (size_t)n * H * W * Cin;
        float* outb = output + (size_t)n * OH * OW * Cout;
        for (int g = 0; g < groups; ++g) {
            const float* wg = weight + (size_t)g * wg_stride;
            const float* bg = bias ? bias + (size_t)g * Cout_p : nullptr;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    size_t opix = ((size_t)oh * OW + ow) * Cout + (size_t)g * Cout_p;
                    for (int oc = 0; oc < Cout_p; ++oc) {
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

/* 调度器: 按每组输出通道数与 CPU 特性选择向量宽度。
 * Cout_per 是向量化维度 (权重/输出在该维连续)。AVX-512 需 Cout_per>=16 才用满 zmm;
 * 否则退到 AVX2 (Cout_per>=8, 8 路 ymm); 更窄或不支持 SIMD 走标量。 */
void acfun_conv2d_grouped(const float* input, int N, int H, int W, int Cin,
                          const float* weight, const float* bias,
                          int KH, int KW, int Cout, int stride, int pad,
                          int groups, float* output, int act) {
    if (groups <= 1 || KH <= 0 || KW <= 0) return;
    int Cout_p = Cout / groups;
#if defined(__GNUC__)
    if (Cout_p >= 16 && __builtin_cpu_supports("avx512f")) {
        acfun_group_conv_avx512(input, N, H, W, Cin, weight, bias, KH, KW, Cout,
                                stride, pad, groups, output, act);
        return;
    }
    if (Cout_p >= 8 && __builtin_cpu_supports("avx2")) {
        acfun_group_conv_avx2(input, N, H, W, Cin, weight, bias, KH, KW, Cout,
                              stride, pad, groups, output, act);
        return;
    }
#endif
    acfun_group_conv_scalar(input, N, H, W, Cin, weight, bias, KH, KW, Cout,
                            stride, pad, groups, output, act);
}
