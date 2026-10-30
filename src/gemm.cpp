/* Acfun GEMM 调度器：运行时按 CPU 特性选择后端，并控制线程数。
 * 本编译单元以通用指令集编译；SIMD 后端在各自带 -march 的单元中实现，
 * 仅在检测到对应 CPU 特性时才调用，避免非法指令。 */
#include "acfun.h"
#include "internal.h"
#include <omp.h>

static int g_threads = 0; /* 0 = 全部核 */
static int g_backend = 0; /* 1=AVX512 2=AVX2 3=Scalar, 0=未探测 */

void acfun_set_threads(int n) { g_threads = n > 0 ? n : 0; }

static int detect_backend(void) {
    if (g_backend) return g_backend;
#if defined(__GNUC__)
    if (__builtin_cpu_supports("avx512f"))      g_backend = 1;
    else if (__builtin_cpu_supports("avx2"))    g_backend = 2;
    else                                        g_backend = 3;
#else
    g_backend = 3;
#endif
    return g_backend;
}

int acfun_backend(void) { return detect_backend(); }

const char* acfun_backend_name(void) {
    int b = detect_backend();
    if (b == 1) return "AVX512";
    if (b == 2) return "AVX2";
    return "Scalar";
}

void acfun_gemm(int M, int N, int K,
                const float* A, const float* B, float* C) {
    int nt = (g_threads > 0) ? g_threads : omp_get_max_threads();
    /* 小矩阵走串行：避免 OpenMP 线程池开销压垮微薄的计算量 */
    if (M < 128 || N < 128 || K < 128) nt = 1;
    omp_set_num_threads(nt);
    int b = detect_backend();
    if (b == 1)      acfun_gemm_avx512(M, N, K, A, B, C);
    else if (b == 2) acfun_gemm_avx2(M, N, K, A, B, C);
    else             acfun_gemm_ref(M, N, K, A, B, C);
}
