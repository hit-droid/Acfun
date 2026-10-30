/* Acfun 推理层实现。卷积走 im2col + GEMM，与 MNN 的 Conv 优化路线一致。 */
#include "acfun.h"
#include <cmath>
#include <cstring>
#include <algorithm>

void acfun_relu(float* d, size_t len) {
    for (size_t i = 0; i < len; ++i) if (d[i] < 0.0f) d[i] = 0.0f;
}
void acfun_relu6(float* d, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        float v = d[i]; if (v < 0.0f) v = 0.0f; else if (v > 6.0f) v = 6.0f; d[i] = v;
    }
}

void acfun_fc(const float* input, int N, int Cin,
              const float* weight, const float* bias, int Cout, float* output) {
    acfun_gemm(N, Cout, Cin, input, weight, output);
    if (bias) {
        for (int n = 0; n < N; ++n)
            for (int c = 0; c < Cout; ++c)
                output[(size_t)n * Cout + c] += bias[c];
    }
}

/* im2col: 把 NHWC 输入的一块卷积展开成 [OH*OW, KH*KW*Cin] 的矩阵 */
static void im2col(const float* in, int H, int W, int Cin,
                   int KH, int KW, int stride, int pad,
                   int OH, int OW, float* col) {
    int ic = KH * KW * Cin;
    std::memset(col, 0, (size_t)OH * OW * ic * sizeof(float));
    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
            float* dst = col + ((size_t)oh * OW + ow) * ic;
            for (int kh = 0; kh < KH; ++kh) {
                int ih = oh * stride - pad + kh;
                for (int kw = 0; kw < KW; ++kw) {
                    int iw = ow * stride - pad + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        std::memcpy(dst, in + ((size_t)ih * W + iw) * Cin, Cin * sizeof(float));
                    dst += Cin;   /* 越界(填充)位置保持 0, 但必须前进以维持列对齐 */
                }
            }
        }
    }
}

void acfun_conv2d(const float* input, int N, int H, int W, int Cin,
                  const float* weight, const float* bias,
                  int KH, int KW, int Cout, int stride, int pad,
                  float* output) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    int ic = KH * KW * Cin;
    float* col = (float*)malloc((size_t)OH * OW * ic * sizeof(float));
    float* gemm_out = (float*)malloc((size_t)OH * OW * Cout * sizeof(float));
    if (!col || !gemm_out) { free(col); free(gemm_out); return; }

    for (int n = 0; n < N; ++n) {
        const float* x = input + (size_t)n * H * W * Cin;
        im2col(x, H, W, Cin, KH, KW, stride, pad, OH, OW, col);
        /* col[OH*OW, ic] * weight[ic, Cout] -> gemm_out[OH*OW, Cout] */
        acfun_gemm(OH * OW, Cout, ic, col, weight, gemm_out);
        if (bias) {
            for (int r = 0; r < OH * OW; ++r)
                for (int c = 0; c < Cout; ++c)
                    gemm_out[(size_t)r * Cout + c] += bias[c];
        }
        std::memcpy(output + (size_t)n * OH * OW * Cout,
                    gemm_out, (size_t)OH * OW * Cout * sizeof(float));
    }
    free(col);
    free(gemm_out);
}

void acfun_maxpool(const float* in, int N, int H, int W, int C,
                   int pool, int stride, int pad, float* out) {
    int OH = (H + 2 * pad - pool) / stride + 1;
    int OW = (W + 2 * pad - pool) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    for (int n = 0; n < N; ++n) {
        for (int oh = 0; oh < OH; ++oh) {
            for (int ow = 0; ow < OW; ++ow) {
                for (int c = 0; c < C; ++c) {
                    float mx = -1e30f;
                    for (int ph = 0; ph < pool; ++ph) {
                        int ih = oh * stride - pad + ph;
                        for (int pw = 0; pw < pool; ++pw) {
                            int iw = ow * stride - pad + pw;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            float v = in[((size_t)n * H + ih) * W * C + iw * C + c];
                            if (v > mx) mx = v;
                        }
                    }
                    out[((size_t)n * OH + oh) * OW * C + ow * C + c] = mx;
                }
            }
        }
    }
}

void acfun_softmax(const float* in, int rows, int cols, float* out) {
    for (int r = 0; r < rows; ++r) {
        const float* ri = in + (size_t)r * cols;
        float* ro = out + (size_t)r * cols;
        float mx = ri[0];
        for (int c = 1; c < cols; ++c) if (ri[c] > mx) mx = ri[c];
        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) { float e = expf(ri[c] - mx); ro[c] = e; sum += e; }
        float inv = 1.0f / sum;
        for (int c = 0; c < cols; ++c) ro[c] *= inv;
    }
}

void acfun_batchnorm(float* d, int size, int channels,
                     const float* mean, const float* var,
                     const float* gamma, const float* beta, float eps) {
    int spatial = size / channels;
    for (int c = 0; c < channels; ++c) {
        float scale = gamma ? gamma[c] : 1.0f;
        float shift = beta ? beta[c] : 0.0f;
        float inv = scale / sqrtf(var[c] + eps);
        for (int s = 0; s < spatial; ++s)
            d[(size_t)s * channels + c] = (d[(size_t)s * channels + c] - mean[c]) * inv + shift;
    }
}
