/* Acfun 1x1 INT8 逐点卷积 — 调度器 + 标量兜底 (v1.8.0)。
 *
 * 背景: 1x1/stride1/pad0 卷积本质是 A[M,K]×B[K,N] 的 GEMM (M=OH*OW, K=Cin, N=Cout)。
 * v1.7.0 之前它复用通用 INT8 GEMM (im2col 零拷贝 + packA/packB + 三级分块),
 * 但 K 很小 (<=64) 时 packA/packB 的扩展 4x4 布局 + 双缓冲 malloc + 微内核
 * 展开初始化固定开销远超点积本身 —— 28²x64 k1 s1 (M=784,K=64,N=64) 实测
 * acfun_gemm_int8_epi 独占 0.396 ms, 而 float 1x1 仅 0.117 ms (INT8 反慢 3.4x)。
 * v1.7.0 的小 K 直接内核虽能绕过打包, 但为通用 3x3 设计的 tap 解码 (每 tap 3 次
 * 数组访问 + 边界判断 + 逐字节移位) 在 1x1 上纯属冗余, 反导致 0.71 ms 回退,
 * 故当时排除 1x1。本模块针对 1x1 特化: A 行连续 4 字节一次加载 (无 tap/边界),
 * B 预打包 4k x 16ch 跨全部 M 行复用, 融合反量化/bias/ReLU 一遍写 float。
 *
 * 数值严格对齐通用 GEMM epilogue: acc = Σ_k inQ[k]·wq[k,j] (k 升序累加);
 * out = acc·alpha[j] + beta[j] (+ReLU)。alpha/beta 已含 scale 与零点校正, 由
 * conv_int8.cpp 预先算好、与 GEMM 路径共用, 故逐元素 max|Δ|=0。 */
#include "acfun.h"
#include "internal.h"
#include <cstddef>

/* 标量兜底: 逐行逐通道 k 升序累加, 与 acfun_gemm_int8_epi_ref 同序同舍入。 */
void acfun_conv1x1_int8_scalar(const uint8_t* inQ, int M, int K, int Cout,
                               const int8_t* wq, const float* alpha, const float* beta,
                               float* output, int act) {
    for (int m = 0; m < M; ++m) {
        const uint8_t* a = inQ + (size_t)m * K;
        float* c = output + (size_t)m * Cout;
        for (int j = 0; j < Cout; ++j) {
            int acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (int)a[k] * (int)wq[(size_t)k * Cout + j];
            float v = (float)acc * alpha[j] + beta[j];
            if (act && v < 0.0f) v = 0.0f;
            c[j] = v;
        }
    }
}

/* 派发: 仅在检测到 avx512vnni 时进向量后端, 否则标量兜底 (零回归可移植)。 */
void acfun_conv1x1_int8(const uint8_t* inQ, int M, int K, int Cout,
                        const int8_t* wq, const float* alpha, const float* beta,
                        float* output, int act) {
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx512vnni")) {
        acfun_conv1x1_int8_avx512(inQ, M, K, Cout, wq, alpha, beta, output, act);
        return;
    }
#endif
    acfun_conv1x1_int8_scalar(inQ, M, K, Cout, wq, alpha, beta, output, act);
}
