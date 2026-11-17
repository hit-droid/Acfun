/* Acfun INT8 卷积激活量化 —— AVX2 后端 (v1.6.0)。
 * 编译单元需带 -mavx2 -mfma, 由 int8_quantize.cpp 在无 AVX-512 时派发。
 * 一次 8 路 float→int→uint8, 尾部标量补齐。 */
#include "internal.h"
#include <immintrin.h>
#include <cstdint>

void acfun_quantize_u8_avx2(const float* in, uint8_t* out, size_t n, float inv_s, int zp) {
    const __m256 vinv = _mm256_set1_ps(inv_s);
    const __m256i vzp = _mm256_set1_epi32(zp);
    const __m256i vzero = _mm256_setzero_si256();
    const __m256i v255 = _mm256_set1_epi32(255);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 x = _mm256_mul_ps(_mm256_loadu_ps(in + i), vinv);
        __m256i q = _mm256_cvtps_epi32(x);
        q = _mm256_add_epi32(q, vzp);
        q = _mm256_max_epi32(q, vzero);
        q = _mm256_min_epi32(q, v255);
        __m128i lo = _mm256_castsi256_si128(q);
        __m128i hi = _mm256_extracti128_si256(q, 1);
        __m128i p16 = _mm_packus_epi32(lo, hi);      /* 8×uint16 */
        __m128i p8 = _mm_packus_epi16(p16, p16);     /* 低 8 字节为结果 */
        _mm_storel_epi64((__m128i*)(out + i), p8);
    }
    if (i < n) {
        alignas(32) float t[8] = {0};
        for (size_t k = 0; k < n - i; ++k) t[k] = in[i + k];
        __m256 x = _mm256_mul_ps(_mm256_loadu_ps(t), vinv);
        __m256i q = _mm256_cvtps_epi32(x);
        q = _mm256_add_epi32(q, vzp);
        q = _mm256_max_epi32(q, vzero);
        q = _mm256_min_epi32(q, v255);
        alignas(32) int32_t o[8];
        _mm256_store_si256((__m256i*)o, q);
        for (size_t k = 0; k < n - i; ++k) out[i + k] = (uint8_t)o[k];
    }
}
