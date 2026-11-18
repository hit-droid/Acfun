/* Acfun 小 K INT8 直接卷积 — 调度器 + 标量兜底 (v1.7.0)。
 *
 * 背景: 现有 INT8 卷积走 im2col(量化) + packA + VNNI GEMM。当 K=KH*KW*Cin 很小
 * (1x1 / 极小 Cin / stride>1) 而 M=OH*OW 很大时, im2col 把输入放大 KH*KW 倍、
 * packA 再放大 4 倍, 两段固定开销远超实际点积, 导致 INT8 反而慢于 float 卷积
 * (224²×3→16 仅 0.2×)。本模块直接在 NHWC uint8 输入上按输出像素计算点积,
 * 无中间缓冲, 把小 K 场景拉回正收益。
 *
 * 数值严格对齐现有 GEMM epilogue 路径:
 *   out = (Σ_k inQ[k]·wq[k,j])·alpha[j] + beta[j]  (+ReLU)
 * 其中 alpha[j]=sa·sw[j], beta[j]=bias[j]-alpha[j]·zp·Sw[j] 由 conv_int8.cpp 预好;
 * padding 位置填 zp=128 (而非 0), 使 (xq-zp)=0 精确还原 float 的 zero padding。 */
#include "acfun.h"
#include "internal.h"
#include <cstddef>
#include <algorithm>

/* 标量兜底: 逐输出像素 / 逐通道直接累加。既是无 VNNI 主机的落地点, 也是本路径的
 * 正确性基线 (与 AVX-512 后端逐元素 max|Δ|=0)。 */
void acfun_conv2d_int8_direct_scalar(const uint8_t* inQ, int N, int H, int W, int Cin,
                                     const int8_t* wq, const float* alpha, const float* beta,
                                     int KH, int KW, int Cout, int stride, int pad,
                                     int OH, int OW, float* output, int act) {
    const int ZP = 128;
    const int K = KH * KW * Cin;
    for (int n = 0; n < N; ++n) {
        const uint8_t* inb = inQ + (size_t)n * H * W * Cin;
        float* outb = output + (size_t)n * OH * OW * Cout;
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                float* dst = outb + ((size_t)oh * OW + ow) * Cout;
                for (int j = 0; j < Cout; ++j) {
                    int acc = 0;
                    /* 必须遍历全部 K 个 tap: 越界 padding 位置以 zp=128 参与累加。
                     * 因 beta 已含 -alpha·zp·Sw[j] (Sw 覆盖全部 K 权重), 只有让 pad 位
                     * 贡献 zp·wq 才能精确抵消、还原 float 的 zero padding; 跳过会偏差。 */
                    for (int kh = 0; kh < KH; ++kh) {
                        int ih = oh * stride - pad + kh;
                        bool ih_ok = (ih >= 0 && ih < H);
                        for (int kw = 0; kw < KW; ++kw) {
                            int iw = ow * stride - pad + kw;
                            bool ok = ih_ok && (iw >= 0 && iw < W);
                            const uint8_t* ip = ok ? inb + ((size_t)ih * W + iw) * Cin : nullptr;
                            const int8_t*  wp = wq + (size_t)((kh * KW + kw) * Cin) * Cout + j;
                            for (int ic = 0; ic < Cin; ++ic) {
                                int xq = ok ? (int)ip[ic] : ZP;
                                acc += xq * (int)wp[(size_t)ic * Cout];
                            }
                        }
                    }
                    float v = (float)acc * alpha[j] + beta[j];
                    if (act && v < 0.0f) v = 0.0f;
                    dst[j] = v;
                }
            }
        }
    }
    (void)ZP; (void)K;
}

/* 派发: 仅在检测到 avx512vnni 时进向量后端, 否则标量兜底 (零回归可移植)。 */
void acfun_conv2d_int8_direct(const uint8_t* inQ, int N, int H, int W, int Cin,
                              const int8_t* wq, const float* alpha, const float* beta,
                              int KH, int KW, int Cout, int stride, int pad,
                              int OH, int OW, float* output, int act) {
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx512vnni")) {
        acfun_conv2d_int8_direct_avx512(inQ, N, H, W, Cin, wq, alpha, beta,
                                        KH, KW, Cout, stride, pad, OH, OW, output, act);
        return;
    }
#endif
    acfun_conv2d_int8_direct_scalar(inQ, N, H, W, Cin, wq, alpha, beta,
                                    KH, KW, Cout, stride, pad, OH, OW, output, act);
}
