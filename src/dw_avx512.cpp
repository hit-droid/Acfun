/* Acfun 深度(逐通道)卷积 — AVX-512F+FMA 后端。
 *
 * 深度卷积: 每个输出通道 c 仅与其同名输入通道 c 做 KH×KW 点积 (groups=Cin=Cout),
 * 权重布局 [C, KH, KW] (w[c*KH*KW + kh*KW + kw])。相较把深度卷积当普通卷积走
 * im2col+GEMM (K=KH*KW 极小, 算术强度崩溃、im2col 还产生 K 倍拷贝), 本路径把输入
 * 跨 16 通道向量化:
 *   - 以 C 维 16 路分块, 每个通道块读取 16 个连续通道 (NHWC 最末维连续, 访存友好);
 *   - 每个输出像素做 KH*KW 次 16 路 FMA, 权重向量预转置进连续缓冲 wT[KH*KW][16],
 *     被该通道块的全部像素复用, 仅 9×16=144 B 权重/块且常驻 L1;
 *   - 边界像素 (padding 区) 用掩码零化 (AVX-512 掩码加载对屏蔽位不触发缺页), 主体像素
 *     走无检查快路径, 分支可完美预测。
 * 这是 MobileNet/EfficientNet 等轻量网络的核心算子, 与 MNN 的深度卷积路径一致。
 * 编译单元需带 -mavx512f -mfma。 */
#include "internal.h"
#include <immintrin.h>
#include <cstring>
#include <cstdlib>
#include <cmath>

void acfun_conv2d_dw_avx512(const float* input, int N, int H, int W, int C,
                            const float* weight, const float* bias,
                            int KH, int KW, int stride, int pad,
                            float* output, int act) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    const int VL = 16;                 /* AVX-512 通道数 */
    int K = KH * KW;
    /* 每通道块预转置权重: wT[k*VL + i] = 通道 (c0+i) 在点 k 处的权重, 16 路连续便于 loadu */
    float wT[16 * 16];                /* K 最大 16×16=256, 取上限; 实际仅 K*VL 有效 */

    for (int c0 = 0; c0 < C; c0 += VL) {
        int cblk = (c0 + VL <= C) ? VL : (C - c0);
        bool full = (cblk == VL);
        __mmask16 m = (__mmask16)((1u << cblk) - 1);
        /* 预转置该通道块的权重 (KH*KW 位置 × 16 通道) */
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < cblk; ++i)
                wT[k * VL + i] = weight[((size_t)(c0 + i)) * K + k];

        for (int n = 0; n < N; ++n) {
            const float* in  = input  + (size_t)n * H * W * C;
            float*       out = output + (size_t)n * OH * OW * C;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    __m512 acc = full
                        ? (bias ? _mm512_loadu_ps(bias + c0) : _mm512_setzero_ps())
                        : (bias ? _mm512_maskz_loadu_ps(m, bias + c0) : _mm512_setzero_ps());
                    for (int kh = 0; kh < KH; ++kh) {
                        for (int kw = 0; kw < KW; ++kw) {
                            int k  = kh * KW + kw;
                            int ih = oh * stride - pad + kh;
                            int iw = ow * stride - pad + kw;
                            __m512 vin;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                const float* p = in + ((size_t)ih * W + iw) * C + c0;
                                vin = full ? _mm512_loadu_ps(p) : _mm512_maskz_loadu_ps(m, p);
                            } else {
                                vin = _mm512_setzero_ps();
                            }
                            __m512 wv = full ? _mm512_loadu_ps(wT + k * VL)
                                             : _mm512_maskz_loadu_ps(m, wT + k * VL);
                            acc = _mm512_fmadd_ps(vin, wv, acc);
                        }
                    }
                    if (act == 1) acc = _mm512_max_ps(acc, _mm512_setzero_ps());
                    float* op = out + ((size_t)oh * OW + ow) * C + c0;
                    if (full) _mm512_storeu_ps(op, acc);
                    else      _mm512_mask_storeu_ps(op, m, acc);
                }
            }
        }
    }
}
