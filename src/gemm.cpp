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

/* 生效核心数: 物理核数 与 cgroup CPU 配额的较小值。cgroup v2 读 cpu.max
 * ("quota period", "max" 表示不限), cgroup v1 读 cpu.cfs_quota_us/period_us。
 * 受限容器(如本机 4 核配额)下把线程上限压到配额内, 避免争抢引发的切换风暴。 */
int acfun_effective_cores(void) {
    int phys = physical_cores();
    long long quota = 0;   /* 0 = 不限 */
    FILE* f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char buf[64];
        if (fgets(buf, sizeof(buf), f) && strncmp(buf, "max", 3) != 0) {
            long long q = 0, p = 0;
            if (sscanf(buf, "%lld %lld", &q, &p) == 2 && p > 0)
                quota = (q + p - 1) / p;
        }
        fclose(f);
    } else {
        f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r");
        long long q = 0;
        if (f) { if (fscanf(f, "%lld", &q) == 1 && q > 0) quota = q; fclose(f); }
        if (quota > 0) {
            f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r");
            long long p = 0;
            if (f) { if (fscanf(f, "%lld", &p) == 1 && p > 0) quota = (quota + p - 1) / p; fclose(f); }
        }
    }
    int eff = (quota > 0 && quota < (long long)phys) ? (int)quota : phys;
    return eff > 0 ? eff : 1;
}

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

    /* cgroup 感知 (v0.9.0): 受限容器 (如本沙箱 cpu.max=4 核) 下, 报告的逻辑/物理核数远超实际
     * 可用 CPU 时间配额。若按 16 线程派发会严重超额订阅 —— 线程在配额上限下互相抢占、上下文
     * 切换风暴使 8T 后吞吐不升反降 (实测 16T 仅 ~15% 线性效率)。必须压到配额内。这正是对标
     * MNN 的"按可用算力选线程"行为: 用满但不超额, 才拿到近线性多核扩展。 */
    int eff = acfun_effective_cores();
    if (nt > eff) nt = eff;

    /* 工作感知线程选择 —— 比单纯 "三维皆<256 才串行" 更稳健:
     * 卷积类 GEMM 虽 M(像素数) 极大、但 N(输出通道, 如 16) 极小, 计算量可能
     * 仅几十 MFLOP。此时 OpenMP 线程派发 + 全 K 面板池(可达数 MB)的分配/零初始化
     * 开销远超计算, 多线程反而更慢 (实测 3x3/112²/16 卷积被拖慢数十倍)。
     * 故按总 FLOP 设每线程下限 (~64M), 不足则回落串行流水分块路径。 */
    long long flop = (long long)2 * M * N * K;
    bool tiny = (M < 256 && N < 256 && K < 256);
    if (tiny) {
        nt = 1;                       /* 极小 GEMM: 派发/池开销远超计算, 直接串行 */
    } else {
        long long cap = flop / (64LL * 1024 * 1024);  /* ~64M FLOP/线程 */
        if (cap < 1) cap = 1;
        if ((long long)nt > cap) nt = (int)cap;
    }
    if (nt > 1) omp_set_num_threads(nt);
    acfun_set_num_threads_runtime(nt);
    int b = detect_backend();
    if (b == 1)      acfun_gemm_avx512(M, N, K, A, B, C);
    else if (b == 2) acfun_gemm_avx2(M, N, K, A, B, C);
    else             acfun_gemm_ref(M, N, K, A, B, C);
}
