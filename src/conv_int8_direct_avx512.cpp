/* Acfun 小 K INT8 直接卷积 — AVX-512 VNNI 后端 (v1.7.0, v1.9.0 多线程并行)。
 * 编译单元需带 -mavx512f -mfma -mavx512vnni; 由 conv_int8_direct.cpp 在检测到
 * avx512vnni 时派发, 否则回落标量版。
 *
 * v1.9.0: 按输出行 (n,oh) 展平做 OpenMP 静态并行。每个 (n,oh,ow) 像素的输出区
 * 互不相交, 权重预打包 Bp / tap 解码表 / alpha / beta 全程只读共享, 累加器为
 * 栈上私有 —— 零写冲突、结果与线程数严格无关。工作感知: 输出元素过少或行数
 * 不足时回落串行, 避免 fork/join 开销压垮小图 (如 56²×16 k1s2)。线程数由
 * acfun_get_num_threads_runtime() 下发, 与 v1.8.0 1x1 内核 / GEMM 范式一致。
 *
 * VNNI 映射 (关键): vpdpbusd 的一个 64B zmm 有 16 个 32-bit lane, lane ℓ 累加
 *   Σ_{t=0..3} A[4ℓ+t](uint8) · B[4ℓ+t](int8)。
 * 直接卷积中, 同一输出像素的 4 个连续 k 输入值 xq[k0..k0+3] 对全部 Cout 通道共享,
 * 故把 A 的全部 16 个 lane 都置为这 4 个字节 (_mm512_broadcast_i32x4), 而让
 * B 的 lane ℓ 承载「输出通道 j0+ℓ 的 4 个连续 k 权重」。一条 dpbusd 即算完
 * 16 通道 × 4 个 k, 无 gather、无 im2col、无 packA。
 *
 * 权重预打包: Bp[cb,Kg,16,4] = wq[(kg*4+t)*Cout + cb*16+lane], 一次生成, 跨
 * 所有输出像素 (M 可达数万) 复用; K、Cout 尾部以 0 填充, 不影响点积。
 * padding 位置的输入 xq 置 zp=128 (而非跳过), 与现有 im2col+epilogue 严格等价。 */
#include "internal.h"
#include <immintrin.h>
#include <omp.h>
#include <cstddef>
#include <vector>

void acfun_conv2d_int8_direct_avx512(const uint8_t* inQ, int N, int H, int W, int Cin,
                                     const int8_t* wq, const float* alpha, const float* beta,
                                     int KH, int KW, int Cout, int stride, int pad,
                                     int OH, int OW, float* output, int act) {
    const int ZP = 128;
    const int VL = 16;
    const int K = KH * KW * Cin;
    const int Kg = (K + 3) / 4;              /* 4-k 组数 (尾部补零) */
    const int CB = (Cout + VL - 1) / VL;     /* 16-通道组数 */

    /* 权重预打包为 VNNI 布局: 每组 = 16 lanes × 4 字节 = 64 字节。
     * 字节 (lane*4+t) = wq[(kg*4+t)*Cout + (cb*16+lane)]。 */
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

    /* tap 预解码表: k -> (通道内偏移 ic, 相对 oh 的输入行偏移 doh, 相对 ow 的列偏移 dow, 越界标志)。
     * ih = oh*stride + doh, iw = ow*stride + dow。整张图所有像素共享, 一次生成;
     * 消除内核最内层的 3 次整数除法 (k/Cin, /KW, %KW) —— 在 K 组×M 像素规模下是头号标量瓶颈。 */
    int tap_ic[64], tap_doh[64], tap_dow[64];
    for (int k = 0; k < K; ++k) {
        int khkw = k / Cin;
        tap_ic[k]  = k - khkw * Cin;
        int kh = khkw / KW;
        int kw = khkw - kh * KW;
        tap_doh[k] = kh - pad;
        tap_dow[k] = kw - pad;
    }

    const __m512 vzero = _mm512_setzero_ps();

    /* v1.9.0: 处理单个输出行 (n, oh) 的全部 ow / 通道块。每行写入互不相交的
     * 输出区, Bp / tap 表 / alpha / beta 只读共享, acc 为栈上局部 —— 线程安全,
     * 结果与线程数严格无关 (逐元素 max|Δ|=0)。 */
    auto process_row = [&](int r) {
        const int n  = r / OH;
        const int oh = r - n * OH;
        const uint8_t* inb = inQ + (size_t)n * H * W * Cin;
        float* outb = output + (size_t)n * OH * OW * Cout;
        for (int ow = 0; ow < OW; ++ow) {
            float* dst = outb + ((size_t)oh * OW + ow) * Cout;
            for (int cb = 0; cb < CB; ++cb) {
                const int j0 = cb * VL;
                const int valid = std::min(VL, Cout - j0);
                const __mmask16 m = (valid == VL) ? 0xFFFFu : (uint16_t)((1u << valid) - 1);

                __m512i acc = _mm512_setzero_si512();
                for (int kg = 0; kg < Kg; ++kg) {
                    /* 由预解码表生成当前像素的 4 个连续 k 输入值, 打包成 32-bit 后广播 16 lanes */
                    uint32_t a4 = 0;
                    for (int t = 0; t < 4; ++t) {
                        int k = kg * 4 + t;
                        int xq = ZP;   /* k>=K (补零组) 或越界 padding 均取 zp */
                        if (k < K) {
                            int ih = oh * stride + tap_doh[k];
                            int iw = ow * stride + tap_dow[k];
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                xq = (int)inb[((size_t)ih * W + iw) * Cin + tap_ic[k]];
                        }
                        a4 |= (uint32_t)xq << (8 * t);
                    }
                    /* 先把 4 字节填满一个 128 位 (4 个 dword 均为 a4), 再广播到 512 位;
                     * 若用 _mm_cvtsi32_si128 则高 96 位为 0, 只有 lane 0-3 有输入。 */
                    __m512i A = _mm512_broadcast_i32x4(_mm_set1_epi32((int)a4));
                    __m512i B = _mm512_loadu_si512((const __m512i*)
                                    (Bp.data() + ((size_t)cb * Kg + kg) * 64));
                    acc = _mm512_dpbusd_epi32(acc, A, B);
                }

                /* epilogue: acc(int32) → float, ×alpha+beta (+无分支 ReLU)。
                 * 显式 mul+add (非 FMA), 与标量参考舍入顺序一致 (max|Δ|=0)。 */
                __m512 va = _mm512_maskz_loadu_ps(m, alpha + j0);
                __m512 vb = _mm512_maskz_loadu_ps(m, beta  + j0);
                __m512 vf = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc), va), vb);
                if (act) vf = _mm512_max_ps(vf, vzero);
                _mm512_mask_storeu_ps(dst + j0, m, vf);
            }
        }
    };

    /* 工作感知线程选择: 大 M (如 224²×3, OH=224 → 224 行) 充分并行; 小图 /
     * 行数不足时回落串行, 避免 fork/join 开销压垮微薄工作量 (如 56²×16 k1s2)。 */
    const int rows = N * OH;
    int nt = acfun_get_num_threads_runtime();
    const long long out_elems = (long long)rows * OW * Cout;
    if (nt > 1 && (rows < nt || out_elems < 65536)) nt = 1;

    if (nt <= 1) {
        for (int r = 0; r < rows; ++r) process_row(r);
    } else {
        #pragma omp parallel num_threads(nt)
        {
            #pragma omp for schedule(static)
            for (int r = 0; r < rows; ++r) process_row(r);
        }
    }
}
