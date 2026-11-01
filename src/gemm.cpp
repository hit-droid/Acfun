/* Acfun GEMM 调度器：运行时按 CPU 特性选择后端，并控制线程数。
 * 默认线程数取物理核数 (避开超线程在 GEMM 上的拖累)，可用 acfun_set_threads()
 * 或环境变量 ACFUN_THREADS 覆盖。本编译单元以通用指令集编译；SIMD 后端在各自
 * 带 -march 的单元中实现，仅在检测到对应 CPU 特性时才调用，避免非法指令。 */
#include "acfun.h"
#include "internal.h"
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <set>
#include <utility>

static int g_threads = 0; /* 0 = 物理核数 */
static int g_backend = 0; /* 1=AVX512 2=AVX2 3=Scalar, 0=未探测 */
static int g_rt_threads = 1; /* 下发给后端的期望线程数 */

void acfun_set_num_threads_runtime(int n) { g_rt_threads = n > 0 ? n : 1; }
int  acfun_get_num_threads_runtime(void)  { return g_rt_threads; }

/* 统计物理核数: 解析 /proc/cpuinfo 中不同的 (physical id, core id) 组合数。
 * 失败回退到 sysconf 的逻辑处理器数。 */
static int physical_cores(void) {
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return (int)sysconf(_SC_NPROCESSORS_ONLN);
    std::set<std::pair<int,int>> ids;
    int phys = -1, core = -1; char line[256];
    while (fgets(line, sizeof(line), f)) {
        if (sscanf(line, "physical id : %d", &phys) == 1) continue;
        if (sscanf(line, "core id : %d", &core) == 1) ids.insert({phys, core});
    }
    fclose(f);
    int n = (int)ids.size();
    return n > 0 ? n : (int)sysconf(_SC_NPROCESSORS_ONLN);
}

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
    /* 线程数: 显式设置优先, 否则 ACFUN_THREADS 环境变量, 否则物理核数 */
    int nt = 0;
    if (g_threads > 0) {
        nt = g_threads;
    } else {
        const char* e = std::getenv("ACFUN_THREADS");
        if (e) { int v = std::atoi(e); if (v > 0) nt = v; }
    }
    if (nt <= 0) nt = physical_cores();
    /* 小矩阵走串行：避免 OpenMP 线程池开销压垮微薄的计算量 */
    if (M < 256 || N < 256 || K < 256) nt = 1;
    if (nt > 1) omp_set_num_threads(nt);
    acfun_set_num_threads_runtime(nt);
    int b = detect_backend();
    if (b == 1)      acfun_gemm_avx512(M, N, K, A, B, C);
    else if (b == 2) acfun_gemm_avx2(M, N, K, A, B, C);
    else             acfun_gemm_ref(M, N, K, A, B, C);
}
