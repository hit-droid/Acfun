/* Acfun 推理层实现。卷积走 im2col + GEMM，与 MNN 的 Conv 优化路线一致。 */
#include "acfun.h"
#include "internal.h"
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

/* 1x1 逐点卷积快路径: 卷积退化为 GEMM, 完全跳过 im2col 的展开/拷贝。
 *   out[p, c] = Σ_d in[p, d] * w[d, c] + bias[c]
 * 当 stride=1 && pad=0 时输入矩阵与输入张量同形, A 直接指向 input, 零拷贝;
 * 仅当 stride>1 (或 pad>0) 才按需做跨步 gather 构造 A。较 im2col+GEMM 省去
 * 一次 [OH*OW, Cin] 的完整展开拷贝, 且算术强度不变、GEMM 内核可直接复用。
 * act=1 在写回时融合 ReLU。 */
static void conv1x1_impl(const float* input, int N, int H, int W, int Cin,
                         const float* weight, const float* bias,
                         int Cout, int stride, int pad, float* output, int act) {
    int OH = (H + 2 * pad - 1) / stride + 1;
    int OW = (W + 2 * pad - 1) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    size_t pixTotal = (size_t)N * OH * OW;       /* 全部 batch 的输出像素数 */
    size_t inPix     = (size_t)N * H * W;

    /* 构造 A 矩阵 [pixTotal, Cin]: stride==1 && pad==0 时与输入同形, 直接复用 */
    const float* Aptr = input;
    float* Abuf = nullptr;
    if (stride != 1 || pad != 0) {
        Abuf = (float*)malloc(pixTotal * Cin * sizeof(float));
        if (!Abuf) return;
        for (size_t p = 0; p < pixTotal; ++p) {
            int n  = (int)(p / ((size_t)OH * OW));
            size_t rp = p % ((size_t)OH * OW);
            int oh = (int)(rp / OW), ow = (int)(rp % OW);
            int ih = oh * stride - pad, iw = ow * stride - pad;
            float* dst = Abuf + p * Cin;
            const float* src = input + (((size_t)n * H + ih) * W + iw) * Cin;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                std::memcpy(dst, src, Cin * sizeof(float));
            else
                std::memset(dst, 0, Cin * sizeof(float));
        }
        Aptr = Abuf;
    }

    float* gemm_out = (float*)malloc(pixTotal * Cout * sizeof(float));
    if (!gemm_out) { free(Abuf); return; }

    /* 整批一次性 GEMM: A[pixTotal, Cin] * W[Cin, Cout] -> gemm_out[pixTotal, Cout] */
    acfun_gemm((int)pixTotal, Cout, Cin, Aptr, weight, gemm_out);

    /* 偏置 + (可选)ReLU 融合, 单次遍历写回 NHWC 输出 */
    for (size_t p = 0; p < pixTotal; ++p) {
        float* yo = output + p * Cout;
        const float* go = gemm_out + p * Cout;
        for (int c = 0; c < Cout; ++c) {
            float v = go[c] + (bias ? bias[c] : 0.0f);
            if (act == 1 && v < 0.0f) v = 0.0f;
            yo[c] = v;
        }
    }
    free(Abuf);
    free(gemm_out);
}

/* 卷积内部实现: act=0 无激活, act=1 融合 ReLU (在输出写入时就地施加, 省去额外扫描) */
static void conv2d_impl(const float* input, int N, int H, int W, int Cin,
                        const float* weight, const float* bias,
                        int KH, int KW, int Cout, int stride, int pad,
                        float* output, int act) {
    /* 1x1 逐点卷积: 直接 GEMM, 跳过 im2col (见 conv1x1_impl) */
    if (KH == 1 && KW == 1) {
        conv1x1_impl(input, N, H, W, Cin, weight, bias, Cout, stride, pad, output, act);
        return;
    }
    /* Winograd F(2,3) 快路径: 仅 3x3 / stride1 / pad1, 且通道与特征图足够大
     * (变换开销固定, 小通道/小图下收益不明显, 回落 im2col+GEMM 更稳)。 */
    if (KH == 3 && KW == 3 && stride == 1 && pad == 1 && H >= 3 && W >= 3
        && Cin >= 4 && Cout >= 4) {
        acfun_winograd_conv3x3(input, N, H, W, Cin, weight, bias, Cout, output, act);
        return;
    }

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
        /* 偏置 + (可选)ReLU 在输出写入时融合, 单次遍历完成 */
        float* yo = output + (size_t)n * OH * OW * Cout;
        for (int r = 0; r < OH * OW; ++r) {
            for (int c = 0; c < Cout; ++c) {
                float v = gemm_out[(size_t)r * Cout + c];
                if (bias) v += bias[c];
                if (act == 1 && v < 0.0f) v = 0.0f;
                yo[(size_t)r * Cout + c] = v;
            }
        }
    }
    free(col);
    free(gemm_out);
}

void acfun_conv2d(const float* input, int N, int H, int W, int Cin,
                  const float* weight, const float* bias,
                  int KH, int KW, int Cout, int stride, int pad,
                  float* output) {
    conv2d_impl(input, N, H, W, Cin, weight, bias, KH, KW, Cout, stride, pad, output, 0);
}

void acfun_conv2d_relu(const float* input, int N, int H, int W, int Cin,
                       const float* weight, const float* bias,
                       int KH, int KW, int Cout, int stride, int pad,
                       float* output) {
    conv2d_impl(input, N, H, W, Cin, weight, bias, KH, KW, Cout, stride, pad, output, 1);
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
