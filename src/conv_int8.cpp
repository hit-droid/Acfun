/* Acfun INT8 量化卷积 (AVX-512 VNNI, 对标 MNN int8 推理路径)。
 *
 * 量化方案 (对称权重 + 逐通道权重 scale + 带零点的逐张量激活, zero-point=128):
 *   - 权重: 每个输出通道 j 一个 scale s_w[j] = max|w[:,j]| / 127; w_q = round(w / s_w[j]) ∈ [-128,127]。
 *   - 激活: 整张输入一个 scale s_a = max|x| / 127, 零点 zp=128; x_q = round(x / s_a) + zp ∈ [0,255] (uint8)。
 *     uint8×int8 的 vpdpbusd 把 A 当无符号数, 故激活必须带上零点才能表达负值: x ≈ s_a·(x_q − zp)。
 * 于是真实卷积 C[i,j] = Σ_k x[k]·w[k,j] ≈ (s_a·s_w[j]) · Σ_k (x_q[k] − zp)·w_q[k,j]
 *   = (s_a·s_w[j]) · ( Σ_k x_q[k]·w_q[k,j] − zp·Σ_k w_q[k,j] ),
 * 后者第二项 −zp·Sw[j] (Sw[j]=Σ_k w_q[k,j]) 作为「零点偏置校正」并入 bias。内积以 vpdpbusd
 * (uint8×int8→int32) 累加, 末了乘 (s_a·s_w[j]) + 校正后 bias (+可选 ReLU) 即得 float 输出。
 * 若直接把负值激活截断到 0 (无零点) 会丢失约一半信号, 误差将接近信号本身体量。
 *
 * 流程: 输入量化 uint8(带零点) -> im2col(量化) -> INT8 GEMM -> int32 -> 反量化(+bias 校正, 可选 ReLU)。
 * 权重离线量化一次 (acfun_quantize_weights), 推理期仅量化激活, 故吞吐由 INT8 GEMM 主导,
 * 较 float 路径带宽省 4×、算力密度升 16×。 */
#include "acfun.h"
#include "internal.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

/* 逐通道对称量化权重为 int8, 返回每通道 scale。w 布局 [K, N] row-major。 */
void acfun_quantize_weights(const float* w, int K, int N, float* scales, int8_t* wq) {
    for (int j = 0; j < N; ++j) {
        float maxa = 0.0f;
        for (int k = 0; k < K; ++k) maxa = std::max(maxa, std::fabs(w[(size_t)k * N + j]));
        float s = maxa / 127.0f;
        if (s <= 0.0f) s = 1.0f;
        scales[j] = s;
        for (int k = 0; k < K; ++k) {
            int v = (int)std::lroundf(w[(size_t)k * N + j] / s);
            if (v > 127) v = 127; else if (v < -128) v = -128;
            wq[(size_t)k * N + j] = (int8_t)v;
        }
    }
}

/* im2col 于已量化的 uint8 输入: 把 [N,H,W,Cin] 卷积展开成 [OH*OW, KH*KW*Cin]。
 * 填充区必须填零点 zp=128 (而非 0): 反量化末了以 −zp·Sw 校正, 仅当填充=x_q=zp 时该列贡献恰
 * 好抵消、还原 float 的 padding=0; 若填 0 则每填充列净留 −zp·wq, 污染全部输出 (此前数值错误根因)。 */
static void im2col_u8(const uint8_t* in, int H, int W, int Cin,
                      int KH, int KW, int stride, int pad,
                      int OH, int OW, uint8_t* col) {
    int ic = KH * KW * Cin;
    std::memset(col, (uint8_t)128, (size_t)OH * OW * ic * sizeof(uint8_t));
    for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
            uint8_t* dst = col + ((size_t)oh * OW + ow) * ic;
            for (int kh = 0; kh < KH; ++kh) {
                int ih = oh * stride - pad + kh;
                for (int kw = 0; kw < KW; ++kw) {
                    int iw = ow * stride - pad + kw;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                        std::memcpy(dst, in + ((size_t)ih * W + iw) * Cin, Cin * sizeof(uint8_t));
                    dst += Cin;
                }
            }
        }
    }
}

void acfun_conv2d_int8(const float* input, int N_, int H, int W, int Cin,
                       const int8_t* wq, const float* wscale, const float* bias,
                       int KH, int KW, int Cout, int stride, int pad,
                       float s_in, float* output, int act) {
    int OH = (H + 2 * pad - KH) / stride + 1;
    int OW = (W + 2 * pad - KW) / stride + 1;
    if (OH <= 0 || OW <= 0) return;
    int K = KH * KW * Cin;
    size_t M = (size_t)N_ * OH * OW;

    /* 1) 求激活量化步长 (逐张量对称 scale, 零点 zp=128)。3x3/s1/p1 且通道足够大时直接路由到
     *    INT8 Winograd 快路径 (其内就地量化, 跳过本函数的独立量化遍); 其余路径才在此量化输入。 */
    size_t inN = (size_t)N_ * H * W * Cin;
    float maxa = 0.0f;
    for (size_t i = 0; i < inN; ++i) maxa = std::max(maxa, std::fabs(input[i]));
    if (maxa < 1e-12f) { std::memset(output, 0, M * Cout * sizeof(float)); return; }
    const int ZP = 128;                 /* 激活零点: uint8 中点, 对应 x=0 */
    float inv_s = 127.0f / maxa;       /* s_a = maxa/127, 故 x_q = round(x/s_a) + ZP */
    float sa = maxa / 127.0f;          /* 反量化步长 s_a, 与量化严格一致 */

    /* 3x3 / stride1 / pad1 且通道足够大: 自动路由到 INT8 Winograd 快路径 (对标 MNN int8 卷积),
     * 乘法量减少 2.25x 且跳过 im2col 展开; 本 TU 带 -mavx512f, 仅在检测到 avx512f 时启用
     * (512-bit 整指令); 否则回落 im2col+INT8 GEMM (其自身再做 avx512-vnni 派发)。Winograd
     * 接收 float 输入并在填充缓冲时就地量化, 省去一遍整张输入的独立量化扫描。 */
    if (KH == 3 && KW == 3 && stride == 1 && pad == 1 && H >= 3 && W >= 3
        && Cin >= 4 && Cout >= 4 && __builtin_cpu_supports("avx512f")) {
        acfun_winograd_conv3x3_int8(input, N_, H, W, Cin, wq, wscale, bias, Cout, sa, inv_s, output, act);
        return;
    }

    /* 2) 量化输入张量为 uint8 (仅非 Winograd 路径需要)。v1.6.0: 改用运行时派发的
     *    AVX-512/AVX2 向量量化内核 (标量兜底), 替代原先逐元素标量 lroundf 循环。 */
    std::vector<uint8_t> inQ(inN);
    acfun_quantize_u8(input, inQ.data(), inN, inv_s, ZP);

    /* 反量化参数: alpha[j]=s_a·s_w[j], beta[j]=bias[j]−alpha[j]·zp·Sw[j]。
     * Sw[j]=Σ_k w_q[k,j], −zp·Sw[j] 为零点偏置校正。直接内核与 GEMM epilogue 共用。 */
    std::vector<float> alpha(Cout), beta(Cout);
    for (int j = 0; j < Cout; ++j) {
        float Swj = 0.0f;
        for (int k = 0; k < K; ++k) Swj += (float)wq[(size_t)k * Cout + j];
        float a = sa * wscale[j];
        alpha[j] = a;
        beta[j]  = (bias ? bias[j] : 0.0f) - a * (float)ZP * Swj;
    }

    /* v1.7.0 小 K 直接卷积路由: K=KH*KW*Cin 很小 (1x1 / 极小 Cin / stride>1) 而
     * M 很大时, im2col 把输入放大 KH*KW 倍、packA 再放大 4 倍, 固定开销远超点积,
     * 是 INT8 卷积慢于 float 的回退根因 (224²×3→16 曾 0.2×)。这类场景直接读 NHWC
     * 量化输入、VNNI 逐像素算点积 + 融合 epilogue, 跳过 im2col 与 packA 两段。
     * 3x3/s1/p1 且通道足够大的情形已被上方 Winograd 快路径截获, 不进这里。
     * 1x1/s1/p0 即便 K 小也无 im2col 开销 (A 零拷贝直指 inQ), 原 GEMM 已高效,
     * 排除以免逐像素直接内核反而回退 (28²×64 k1 实测)。阈值 K<=64 精准覆盖其余
     * 回退点 (K=16/27), 大 K 仍走原 GEMM 零回归。 */
    bool zero_copy_1x1 = (KH == 1 && KW == 1 && stride == 1 && pad == 0);
    if (K <= 64 && !zero_copy_1x1) {
        acfun_conv2d_int8_direct(inQ.data(), N_, H, W, Cin, wq, alpha.data(), beta.data(),
                                 KH, KW, Cout, stride, pad, OH, OW, output, act);
        return;
    }

    /* 构造 GEMM 左矩阵 A[M, K]: 1x1/s1/p0 零拷贝 (A 直接指向 inQ), 否则 im2col(量化) */
    const uint8_t* Aptr = inQ.data();
    std::vector<uint8_t> Aq;
    if (!(KH == 1 && KW == 1 && stride == 1 && pad == 0)) {
        Aq.resize(M * K);
        im2col_u8(inQ.data(), H, W, Cin, KH, KW, stride, pad, OH, OW, Aq.data());
        Aptr = Aq.data();
    }

    /* 3)+4) INT8 GEMM + epilogue 融合 (对标 MNN store fusion):
     *    内积 i32[j] = Σ_k x_q[k]·w_q[k,j]; 反量化真实值
     *      = i32[j]·(s_a·s_w[j]) + (bias[j] − s_a·s_w[j]·zp·Sw[j]), 再可选 ReLU。
     *    把 alpha/beta 由 acfun_gemm_int8_epi 在 GEMM 微内核写回阶段就地完成
     *    「×alpha + beta + ReLU」, 直接写 float output。 */
    acfun_gemm_int8_epi((int)M, Cout, K, Aptr, wq, output, alpha.data(), beta.data(), act);
}
