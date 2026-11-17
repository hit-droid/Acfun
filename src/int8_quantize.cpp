/* Acfun INT8 卷积激活量化 —— 调度器 + 标量参考 (v1.6.0)。
 *
 * INT8 卷积路径的量化遍 x_q = clamp(round(x*inv_s)+zp) 是 O(inN) 逐元素操作, 此前为
 * 标量 lroundf 循环。在大特征图 (如 224²×3, inN=150528) 上虽非 GEMM 瓶颈, 但固定开销
 * 可观 (~0.1 ms)。本单元派发 AVX-512/AVX2 向量内核; 标量版作兜底与正确性基线。
 *
 * 数值说明: 标量版保持原 lroundf + clamp 结合顺序; SIMD 版用 cvtps (nearest-even),
 * 仅在极罕见的恰好 .5 tie 上差 1 个量化级, 对反量化精度影响 <1 LSB, 可忽略。 */

#include "acfun.h"
#include "internal.h"
#include <cmath>

static int g_q_backend = 0;
static int detect_q_backend(void) {
    if (g_q_backend) return g_q_backend;
#if defined(__GNUC__)
    /* 主流 avx512f 主机均带 avx512bw (量化 uint8 打包所需, 与 winograd_int8 同假设) */
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"))
        g_q_backend = 1;
    else if (__builtin_cpu_supports("avx2"))
        g_q_backend = 2;
    else
        g_q_backend = 3;
#else
    g_q_backend = 3;
#endif
    return g_q_backend;
}

void acfun_quantize_u8_scalar(const float* in, uint8_t* out, size_t n, float inv_s, int zp) {
    for (size_t i = 0; i < n; ++i) {
        int v = (int)std::lroundf(in[i] * inv_s) + zp;
        out[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
}

void acfun_quantize_u8(const float* in, uint8_t* out, size_t n, float inv_s, int zp) {
    int b = detect_q_backend();
    if (b == 1)      acfun_quantize_u8_avx512(in, out, n, inv_s, zp);
    else if (b == 2) acfun_quantize_u8_avx2(in, out, n, inv_s, zp);
    else             acfun_quantize_u8_scalar(in, out, n, inv_s, zp);
}
