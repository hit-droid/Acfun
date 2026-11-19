/* Acfun 1x1 INT8 逐点卷积 — AVX-512 VNNI 后端 (v1.8.0)。
 * 编译需 -mavx512f -mfma -mavx512vnni -ffp-contract=off; 由 conv1x1_int8.cpp
 * 在检测到 avx512vnni 时派发, 否则回落标量版。
 *
 * VNNI 映射: 1x1 中同一输出像素的 K 个输入对全部 Cout 通道共享。把连续 4 个 k
 * 输入字节 a4 广播到 16 个 lane (每 lane 的 4 字节均为 a4[0..3]), B 的 lane ℓ
 * 承载输出通道 j0+ℓ 在这 4 个 k 上的权重。一条 vpdpbusd 即算完 16 通道 × 4 个 k:
 *   acc[ℓ] += Σ_{t=0..3} a4[t] · wq[(kg*4+t)*Cout + j0+ℓ]
 * 无 im2col、无 packA、无 tap 解码 (1x1 无 padding、无 stride 采样)。
 *
 * 结构: B 预打包一次 (O(K*Cout), 跨 M 行复用); 每行按 kg 遍历, A 的 4 字节只 load/
 * broadcast 一次, 对 CBMAX 个 16 通道块复用; CB 个 int32 累加器跨 kg 常驻寄存器。
 * epilogue 连续 16 通道用掩码 load/store (无通用 GEMM 的 gather + 标量 scatter)。 */
#include "internal.h"
#include <immintrin.h>
#include <cstddef>
#include <cstring>
#include <vector>
#include <algorithm>
#include <omp.h>

void acfun_conv1x1_int8_avx512(const uint8_t* inQ, int M, int K, int Cout,
                               const int8_t* wq, const float* alpha, const float* beta,
                               float* output, int act) {
    if (M <= 0 || K <= 0 || Cout <= 0) return;
    const int VL = 16;
    const int CBMAX = 32;                       /* 常驻累加器上限 (Cout<=512 单块) */
    const int Kg = (K + 3) / 4;                 /* 4-k 组数 (尾部补零) */
    const int CB = (Cout + VL - 1) / VL;        /* 16-通道组数 */

    /* 权重预打包为 VNNI 布局: 组 (cb,kg) = 16 lanes × 4 字节 = 64 字节;
     * 字节 (lane*4+t) = wq[(kg*4+t)*Cout + cb*16+lane], 越界 (k>=K/j>=Cout) 补 0。 */
    std::vector<int8_t> Bp((size_t)CB * Kg * 64, 0);
    for (int cb = 0; cb < CB; ++cb) {
        for (int kg = 0; kg < Kg; ++kg) {
            int8_t* dst = Bp.data() + ((size_t)cb * Kg + kg) * 64;
            for (int lane = 0; lane < VL; ++lane) {
                int j = cb * VL + lane;
                for (int t = 0; t < 4; ++t) {
                    int k = kg * 4 + t;
                    dst[lane * 4 + t] = (j < Cout && k < K)
                                      ? wq[(size_t)k * Cout + j] : 0;
                }
            }
        }
    }

    const __m512 vzero = _mm512_setzero_ps();
    const __m512i vzeroi = _mm512_setzero_si512();

    /* 处理单行: A 零拷贝连续, 按 cb 分块 (每块 CBMAX 个累加器常驻 kg 循环)。 */
    auto process_row = [&](int m) {
        const uint8_t* a = inQ + (size_t)m * K;
        float* orow = output + (size_t)m * Cout;
        for (int cbBase = 0; cbBase < CB; cbBase += CBMAX) {
            int cbN = std::min(CBMAX, CB - cbBase);
            __m512i acc[CBMAX];
            for (int c = 0; c < cbN; ++c) acc[c] = vzeroi;
            for (int kg = 0; kg < Kg; ++kg) {
                /* A 的 4 个连续 k 输入字节: 整组有效则一次 32-bit load, 否则逐字节补零。 */
                uint32_t a4 = 0;
                if ((kg + 1) * 4 <= K) {
                    std::memcpy(&a4, a + (size_t)kg * 4, sizeof(uint32_t));
                } else {
                    uint8_t tmp[4] = {0, 0, 0, 0};
                    for (int t = 0; t < 4 && kg * 4 + t < K; ++t)
                        tmp[t] = a[(size_t)kg * 4 + t];
                    std::memcpy(&a4, tmp, sizeof(uint32_t));
                }
                __m512i Avec = _mm512_broadcast_i32x4(_mm_set1_epi32((int)a4));
                for (int c = 0; c < cbN; ++c) {
                    __m512i Bv = _mm512_loadu_si512((const __m512i*)
                        (Bp.data() + ((size_t)(cbBase + c) * Kg + kg) * 64));
                    acc[c] = _mm512_dpbusd_epi32(acc[c], Avec, Bv);
                }
            }
            /* epilogue: acc(int32)→float, ×alpha+beta (+无分支 ReLU), 连续 16 通道写回。 */
            for (int c = 0; c < cbN; ++c) {
                int j0 = (cbBase + c) * VL;
                int valid = std::min(VL, Cout - j0);
                __mmask16 mk = (valid == VL) ? 0xFFFFu : (uint16_t)((1u << valid) - 1);
                __m512 va = _mm512_maskz_loadu_ps(mk, alpha + j0);
                __m512 vb = _mm512_maskz_loadu_ps(mk, beta + j0);
                __m512 vf = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[c]), va), vb);
                if (act) vf = _mm512_max_ps(vf, vzero);
                _mm512_mask_storeu_ps(orow + j0, mk, vf);
            }
        }
    };

    const int nt = acfun_get_num_threads_runtime();
    if (nt <= 1) {
        for (int m = 0; m < M; ++m) process_row(m);
    } else {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for schedule(static)
            for (int m = 0; m < M; ++m) process_row(m);
        }
    }
}
