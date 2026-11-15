/* Acfun 小通道组分组卷积 — 调度器 + 标量参考。
 * 通用 -O3 编译; AVX-512 内核在单独编译单元中实现, 运行时按 CPU 特性派发。
 *
 * 动机: 一般分组卷积原实现走「逐组抽取 in_g 缓冲 → 递归 conv2d_impl
 * (小通道下 Winograd 变换 / 极小 GEMM 派发的固定开销巨大) → 写回 out_g」三明治,
 * 这些固定开销被 groups 次放大, 在小组 (Cin_per≤8, Cout_per≤16) 下反而比朴素
 * 直乘慢 3–10× (见 benchmark_result.txt 的 g8/g6 用例)。本内核把全部组放进一个
 * 统一循环直接算完: 权重布局 [g][kh][kw][ic][oc] (组内 oc 连续), 输出 NHWC,
 * 逐像素沿 oc 累加 (AVX-512 16 路向量化), 无中间缓冲、无逐组后端派发、无
 * Winograd 变换 —— 对标 MNN 对 group conv 的专属直接路径。 */
#include "acfun.h"
#include "internal.h"
#include <cmath>
#include <cstring>

/* 标量参考 (正确性基线 + 可移植兜底): 权重布局 [g][kh][kw][ic][oc]。 */
void acfun_conv2d_grouped_small_scalar(const float* input, int N, int H, int W, int Cin,
                                       const float* weight, const float* bias,
                                       int KH, int KW, int Cout, int stride, int pad,
                                       int groups, float* output, int act) {
    if (KH <= 0 || KW <= 0 || groups <= 1) return;
    int Cp  = Cin / groups;
    int Cop = Cout / groups;
    if (Cp <= 0 || Cop <= 0 || Cop > 64) return;
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;

    float acc[64];
    for (int g = 0; g < groups; ++g) {
        const float* wg = weight + (size_t)g * Cop * KH * KW * Cp;
        const float* bg = bias ? bias + (size_t)g * Cop : nullptr;
        for (int n = 0; n < N; ++n) {
            const float* in  = input  + (size_t)n * H  * W  * Cin;
            float*       out = output + (size_t)n * OH * OW * Cout;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    for (int oc = 0; oc < Cop; ++oc) acc[oc] = bg ? bg[oc] : 0.0f;
                    for (int kh = 0; kh < KH; ++kh) {
                        int ih = oh * stride - pad + kh;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = ow * stride - pad + kw;
                            if (iw < 0 || iw >= W) continue;
                            const float* inrow = in + ((size_t)ih * W + iw) * Cin + (size_t)g * Cp;
                            const float* wbase = wg + (size_t)(kh * KW + kw) * Cp * Cop;
                            for (int ic = 0; ic < Cp; ++ic) {
                                float a = inrow[ic];
                                const float* wp = wbase + (size_t)ic * Cop;
                                for (int oc = 0; oc < Cop; ++oc) acc[oc] += a * wp[oc];
                            }
                        }
                    }
                    float* op = out + ((size_t)oh * OW + ow) * Cout + (size_t)g * Cop;
                    for (int oc = 0; oc < Cop; ++oc) {
                        float v = acc[oc];
                        if (act == 1 && v < 0.0f) v = 0.0f;
                        op[oc] = v;
                    }
                }
            }
        }
    }
}

/* 调度器: 检测到 avx512f 走 512-bit 向量化, 否则标量。 */
void acfun_conv2d_grouped_small(const float* input, int N, int H, int W, int Cin,
                                const float* weight, const float* bias,
                                int KH, int KW, int Cout, int stride, int pad,
                                int groups, float* output, int act) {
    if (KH <= 0 || KW <= 0 || groups <= 1) return;
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx512f")) {
        acfun_conv2d_grouped_small_avx512(input, N, H, W, Cin, weight, bias,
                                          KH, KW, Cout, stride, pad, groups, output, act);
        return;
    }
#endif
    acfun_conv2d_grouped_small_scalar(input, N, H, W, Cin, weight, bias,
                                      KH, KW, Cout, stride, pad, groups, output, act);
}
