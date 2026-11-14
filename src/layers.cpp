/* Acfun 推理层实现。卷积走 im2col + GEMM，与 MNN 的 Conv 优化路线一致。 */
#include "acfun.h"
#include "internal.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <vector>
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
    /* 偏置经融合后处理 (epilogue) 在 GEMM 写回时就地叠加, 省去一遍独立全张量扫描。
     * FC 层不在此叠 ReLU (ReLU 由模型执行器作为独立层施加), 故 act=0。 */
    acfun_gemm_epi(N, Cout, Cin, input, weight, output, bias, 0);
}

void acfun_fc_relu(const float* input, int N, int Cin,
                   const float* weight, const float* bias, int Cout, float* output) {
    acfun_gemm(N, Cout, Cin, input, weight, output);
    /* 偏置 + 融合 ReLU 单次遍历: 省去 acfun_fc + 独立 acfun_relu 的二次全张量扫描 */
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < Cout; ++c) {
            float v = output[(size_t)n * Cout + c] + (bias ? bias[c] : 0.0f);
            if (v < 0.0f) v = 0.0f;
            output[(size_t)n * Cout + c] = v;
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

    /* 整批一次性 GEMM + 融合后处理 (epilogue): A[pixTotal, Cin] * W[Cin, Cout] -> output[pixTotal, Cout]。
     * NHWC 输出与 GEMM 左矩阵同形 (每像素一行、Cout 列), 故 GEMM 直接写回 output,
     * 偏置与 (可选) ReLU 在 GEMM 写回时就地叠加, 省去独立的 gemm_out 缓冲与全张量扫描。 */
    acfun_gemm_epi((int)pixTotal, Cout, Cin, Aptr, weight, output, bias, act);
    free(Abuf);
}

/* 卷积内部实现: act=0 无激活, act=1 融合 ReLU (在输出写入时就地施加, 省去额外扫描)。
 * groups>1 时进入分组卷积: 深度卷积 (groups==Cin==Cout) 路由到专属深度路径, 否则各组独立
 * 复用引擎既有 1x1 / Winograd / im2col 子路径。 */
static void conv2d_impl(const float* input, int N, int H, int W, int Cin,
                        const float* weight, const float* bias,
                        int KH, int KW, int Cout, int stride, int pad,
                        int groups, float* output, int act) {
    /* ---- 分组卷积 ---- */
    if (groups > 1) {
        if (groups == Cin && groups == Cout) {
            /* 深度可分离卷积: 权重内存布局与 [C,KH,KW] 完全一致 (Cout_per=1 -> 每通道 9 权重) */
            acfun_conv2d_dw(input, N, H, W, Cin, weight, bias, KH, KW, stride, pad, output, act);
            return;
        }
        int Cin_per  = Cin / groups;
        int Cout_per = Cout / groups;
        int OH = (H + 2 * pad - KH) / stride + 1;
        int OW = (W + 2 * pad - KW) / stride + 1;
        if (OH <= 0 || OW <= 0) return;
        /* NHWC 下各组的输入/输出通道在每像素内连续, 但像素间步长 = 总通道数,
         * 无法用单一指针偏移表达, 故每组先抽取连续缓冲再调用引擎子路径,
         * 权重块本就按 [(kh,kw,ic)][oc] 连续存放无需重排。 */
        std::vector<float> in_g((size_t)N * H * W * Cin_per);
        std::vector<float> out_g((size_t)N * OH * OW * Cout_per);
        for (int g = 0; g < groups; ++g) {
            for (size_t p = 0; p < (size_t)N * H * W; ++p) {
                std::memcpy(in_g.data() + p * Cin_per,
                            input + p * Cin + (size_t)g * Cin_per,
                            (size_t)Cin_per * sizeof(float));
            }
            conv2d_impl(in_g.data(), N, H, W, Cin_per,
                        weight + (size_t)g * Cout_per * KH * KW * Cin_per,
                        bias ? bias + g * Cout_per : nullptr,
                        KH, KW, Cout_per, stride, pad, 1,
                        out_g.data(), act);
            for (size_t p = 0; p < (size_t)N * OH * OW; ++p) {
                std::memcpy(output + p * Cout + (size_t)g * Cout_per,
                            out_g.data() + p * Cout_per,
                            (size_t)Cout_per * sizeof(float));
            }
        }
        return;
    }

    /* 1x1 逐点卷积: 直接 GEMM, 跳过 im2col (见 conv1x1_impl) */
    if (KH == 1 && KW == 1) {
        conv1x1_impl(input, N, H, W, Cin, weight, bias, Cout, stride, pad, output, act);
        return;
    }
    /* Winograd 快路径: 仅 3x3 / stride1 / pad1, 且通道与特征图足够大
     * (变换开销固定, 小通道/小图下收益不明显, 回落 im2col+GEMM 更稳)。
     * 瓦片选择 (实测 4 核 Zen5 上 F(4,3) 仅在大图+大通道时胜出):
     *   - F(4,3): 6x6 输入 / 4x4 输出 / 瓦片数降为 F(2,3) 的 1/4, 乘法量/输出降 1.78x,
     *     仅在特征图>=112² 且 Cin,Cout>=64 (计算受限, 算术削减才生效) 时启用; 小图/小通道
     *     下因 NHWC 跨行 gather 与边界瓦片填充浪费反而更慢。
     *   - 其余大图与小图一律 F(2,3) (瓦片更小、边界浪费更低, 且输出变换已 512-bit 向量化)。
     * 这正是 MNN 按张量尺寸在 F(2,3)/F(4,3) 间派发的思路。 */
    if (KH == 3 && KW == 3 && stride == 1 && pad == 1 && H >= 3 && W >= 3
        && Cin >= 4 && Cout >= 4) {
        if (H >= 112 && W >= 112 && Cin >= 64 && Cout >= 64)
            acfun_winograd43_conv3x3(input, N, H, W, Cin, weight, bias, Cout, output, act);
        else
            acfun_winograd_conv3x3(input, N, H, W, Cin, weight, bias, Cout, output, act);
        return;
    }

    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    int ic = KH * KW * Cin;
    float* col = (float*)malloc((size_t)OH * OW * ic * sizeof(float));
    if (!col) return;

    for (int n = 0; n < N; ++n) {
        const float* x = input + (size_t)n * H * W * Cin;
        im2col(x, H, W, Cin, KH, KW, stride, pad, OH, OW, col);
        /* col[OH*OW, ic] * weight[ic, Cout] -> yo[OH*OW, Cout]; 偏置与 (可选) ReLU
         * 经融合后处理 (epilogue) 在 GEMM 写回时就地叠加, 直接落 NHWC 输出,
         * 省去独立的 gemm_out 缓冲 + 一遍全张量扫描 (对标 MNN post-GEMM fusion)。 */
        float* yo = output + (size_t)n * OH * OW * Cout;
        acfun_gemm_epi(OH * OW, Cout, ic, col, weight, yo, bias, act);
    }
    free(col);
}

void acfun_conv2d(const float* input, int N, int H, int W, int Cin,
                  const float* weight, const float* bias,
                  int KH, int KW, int Cout, int stride, int pad,
                  float* output) {
    conv2d_impl(input, N, H, W, Cin, weight, bias, KH, KW, Cout, stride, pad, 1, output, 0);
}

void acfun_conv2d_relu(const float* input, int N, int H, int W, int Cin,
                       const float* weight, const float* bias,
                       int KH, int KW, int Cout, int stride, int pad,
                       float* output) {
    conv2d_impl(input, N, H, W, Cin, weight, bias, KH, KW, Cout, stride, pad, 1, output, 1);
}

void acfun_conv2d_g(const float* input, int N, int H, int W, int Cin,
                    const float* weight, const float* bias,
                    int KH, int KW, int Cout, int stride, int pad, int groups,
                    float* output, int act) {
    conv2d_impl(input, N, H, W, Cin, weight, bias, KH, KW, Cout, stride, pad,
                groups > 1 ? groups : 1, output, act);
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
