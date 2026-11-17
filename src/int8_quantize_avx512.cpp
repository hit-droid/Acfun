/* Acfun INT8 卷积激活量化 —— AVX-512 后端 (v1.6.0)。
 * 编译单元需带 -mavx512f -mfma -mavx512bw, 由 int8_quantize.cpp 在检测到
 * avx512f+bw 时派发。一次 16 路 float→int→uint8, 尾部标量补齐。 */
#include "internal.h"
#include <immintrin.h>
#include <cstdint>

void acfun_quantize_u8_avx512(const float* in, uint8_t* out, size_t n, float inv_s, int zp) {
    const __m512 vinv = _mm512_set1_ps(inv_s);
    const __m512i vzp = _mm512_set1_epi32(zp);
    const __m512i vzero = _mm512_setzero_si512();
    const __m512i v255 = _mm512_set1_epi32(255);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 x = _mm512_mul_ps(_mm512_loadu_ps(in + i), vinv);
        __m512i q = _mm512_cvtps_epi32(x);          /* nearest-even */
        q = _mm512_add_epi32(q, vzp);
        q = _mm512_max_epi32(q, vzero);
        q = _mm512_min_epi32(q, v255);
        __m128i b = _mm512_cvtusepi32_epi8(q);      /* 16 × uint8 (saturate) */
        _mm_storeu_si128((__m128i*)(out + i), b);
    }
    if (i < n) {  /* 尾部 */
        int left = (int)(n - i);
        __mmask16 mask = (__mmask16)((1u << left) - 1u);
        __m512 x = _mm512_maskz_loadu_ps(mask, in + i);
        __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(x, vinv));
        q = _mm512_add_epi32(q, vzp);
        q = _mm512_max_epi32(q, vzero);
        q = _mm512_min_epi32(q, v255);
        alignas(16) int32_t t[16];
        _mm512_store_si512((__m512i*)t, q);
        for (int k = 0; k < left; ++k) out[i + k] = (uint8_t)t[k];
    }
}
