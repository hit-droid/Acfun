/* Acfun 小通道组分组卷积 — AVX-512F+FMA 后端。
 *
 * 与逐组三明治实现不同, 本内核把全部组放进统一循环, 沿组内输出通道 oc 做 16 路
 * 向量化。关键观察:
 *   - 权重布局 [g][kh][kw][ic][oc] 中组内 oc 连续 —— 一次 512-bit load 即取齐
 *     16 个输出通道的权重;
 *   - NHWC 输出中组内 oc 也连续 —— 一次 512-bit store 写回 16 个通道;
 *   - 对固定 (g, 像素, ic), 输入 in[...+g*Cp+ic] 是单个标量, 广播到 16 路 oc
 *     通道作乘子 (vbroadcastss), 一次 FMA 完成 16 个输出通道的累加。
 * 每个 (g, 像素, oc 块) 维护一个 16 路寄存器累加器 acc, 跨 kh/kw/ic 归约,
 * 末了一次写回 (act 融合为无分支 vmax)。全程无 in_g/out_g 中间缓冲、无
 * Winograd 变换、无逐组 GEMM 派发, 把小组卷积的固定开销压到接近零。
 * 编译单元需带 -mavx512f -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <cstddef>

void acfun_conv2d_grouped_small_avx512(const float* input, int N, int H, int W, int Cin,
                                       const float* weight, const float* bias,
                                       int KH, int KW, int Cout, int stride, int pad,
                                       int groups, float* output, int act) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    const int VL = 16;
    int Cp  = Cin / groups;
    int Cop = Cout / groups;
    const __m512 vzero = _mm512_setzero_ps();

    for (int g = 0; g < groups; ++g) {
        const float* wg = weight + (size_t)g * Cop * KH * KW * Cp;
        const float* bg = bias ? bias + (size_t)g * Cop : nullptr;
        for (int n = 0; n < N; ++n) {
            const float* in  = input  + (size_t)n * H  * W  * Cin;
            float*       out = output + (size_t)n * OH * OW * Cout;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    for (int oc0 = 0; oc0 < Cop; oc0 += VL) {
                        int cblk = (oc0 + VL <= Cop) ? VL : (Cop - oc0);
                        bool full = (cblk == VL);
                        __mmask16 m = (__mmask16)((1u << cblk) - 1);
                        __m512 acc = full
                            ? (bg ? _mm512_loadu_ps(bg + oc0) : vzero)
                            : (bg ? _mm512_maskz_loadu_ps(m, bg + oc0) : vzero);
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * stride - pad + kh;
                            if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * stride - pad + kw;
                                if (iw < 0 || iw >= W) continue;
                                const float* inrow = in + ((size_t)ih * W + iw) * Cin
                                                       + (size_t)g * Cp;
                                const float* wbase = wg + (size_t)(kh * KW + kw) * Cp * Cop + oc0;
                                for (int ic = 0; ic < Cp; ++ic) {
                                    __m512 vin = _mm512_set1_ps(inrow[ic]);
                                    const float* wp = wbase + (size_t)ic * Cop;
                                    __m512 wv = full ? _mm512_loadu_ps(wp)
                                                     : _mm512_maskz_loadu_ps(m, wp);
                                    acc = _mm512_fmadd_ps(vin, wv, acc);
                                }
                            }
                        }
                        if (act == 1) acc = _mm512_max_ps(acc, vzero);
                        float* op = out + ((size_t)oh * OW + ow) * Cout
                                        + (size_t)g * Cop + oc0;
                        if (full) _mm512_storeu_ps(op, acc);
                        else      _mm512_mask_storeu_ps(op, m, acc);
                    }
                }
            }
        }
    }
}
