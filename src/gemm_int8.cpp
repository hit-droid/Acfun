/* Acfun INT8 GEMM 调度器：标量参考 + 运行时按 CPU 特性派发到 VNNI 后端。
 * 与本编译单元同级的 gemm_int8_avx512.cpp 携带 -mavx512f -mfma -mavx512vnni,
 * 仅在检测到 avx512vnni 时才调用, 避免非法指令。 */
#include "acfun.h"
#include "internal.h"
#include <cstdlib>
#include <cstring>
#include <set>
#include <utility>

void acfun_gemm_int8_ref_ld(int M, int N, int K,
                            int ldA, int ldB, int ldC,
                            const uint8_t* A, const int8_t* B, int32_t* C) {
    for (int i = 0; i < M; ++i) {
        const uint8_t* a = A + (size_t)i * ldA;
        int32_t* c = C + (size_t)i * ldC;
        for (int j = 0; j < N; ++j) c[j] = 0;
        for (int k = 0; k < K; ++k) {
            int av = (int)a[k];
            if (av == 0) continue;
            const int8_t* b = B + (size_t)k * ldB;
            for (int j = 0; j < N; ++j) c[j] += (int32_t)av * (int32_t)b[j];
        }
    }
}

void acfun_gemm_int8_ref(int M, int N, int K,
                         const uint8_t* A, const int8_t* B, int32_t* C) {
    acfun_gemm_int8_ref_ld(M, N, K, K, N, N, A, B, C);
}

static int g_int8_backend = 0; /* 1=VNNI, 0=未探测 */

static int detect_int8_backend(void) {
    if (g_int8_backend) return g_int8_backend;
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx512vnni")) g_int8_backend = 1;
    else                                       g_int8_backend = 2; /* 标量兜底 (ref) */
#else
    g_int8_backend = 2;
#endif
    return g_int8_backend;
}

void acfun_gemm_int8(int M, int N, int K,
                     const uint8_t* A, const int8_t* B, int32_t* C) {
    int b = detect_int8_backend();
    if (b == 1) acfun_gemm_int8_avx512(M, N, K, A, B, C);
    else        acfun_gemm_int8_ref(M, N, K, A, B, C);
}
