/* Acfun GEMM 自动调参 (autotuning)
 * ---------------------------------------------------------------
 * 分块 GEMM 的 KC(命中 L2 的 K 分块) 与 NC(命中 L2/L1 的 N 分块) 最优值取决于具体
 * 微架构的缓存层级与 FMA 吞吐 —— 纯静态启发式(仅按 L2 容量推导)无法精确匹配真实硬件。
 * 这正是 MNN / OpenBLAS 引入自动调参的核心动机。
 *
 * 本模块对标之:
 *   1. 在目标机实测一组候选 (KC, NC) 网格, 取最快者作为该机的调参结果;
 *   2. 持久化到本地缓存 (~/.cache/acfun/acfun_tune_<host签名>.txt), 下次运行自动读取,
 *      无需重新调参 —— 部署/换机只需 ACFUN_AUTOTUNE=1 执行一次;
 *   3. 内置常见服务端 CPU 的参考调参表, 使同类硬件"开箱即快"; 其余硬件回落静态启发式。
 *
 * 调参只改变分块尺寸(不改变计算结果), 且有完整的尾部余数处理, 故可安全替换启发式路径。
 * 本编译单元以通用指令集(-O3)编译, 通过 acfun_gemm 调度器(运行时派发)实测后端性能。 */
#include "internal.h"
#include "acfun.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <unistd.h>
#include <sys/stat.h>

/* ---------- 候选分块网格 ---------- */
static const int KC_LIST[] = {128, 192, 256, 320, 384, 448, 512};
static const int NC_LIST[] = { 64,  96, 128, 160, 192, 256};
static const int N_KC = (int)(sizeof(KC_LIST) / sizeof(KC_LIST[0]));
static const int N_NC = (int)(sizeof(NC_LIST) / sizeof(NC_LIST[0]));

/* ---------- 参考调参表 (常见服务端 CPU 的保守安全默认值) ----------
 * 命中规则: host 签名(CPU 型号串)包含 key 子串即采用其 (st/mt) KC/NC; 未命中时回落静态启发式。
 * 表中值为"经本机严格无回归门控"得到的安全配置(KC=256 / ST-NC=96 / MT-NC=128):
 * 在常见服务端 CPU 上较纯 L2 启发式(NC=128)持平或小幅更优, 且不会在小矩阵上退化。
 * 真正的最优值由 ACFUN_AUTOTUNE=1 的运行时调参按本机实测得出并持久化到本地缓存。 */
struct RefTune { const char* key; int st_kc, st_nc, mt_kc, mt_nc; };
static const RefTune REF_TUNE[] = {
    {"EPYC 9",        256,  96, 256, 128},
    {"EPYC 7",        256,  96, 256, 128},
    {"Xeon",          256,  96, 256, 128},
    {"Intel(R) Core", 256,  96, 256, 128},
};
static const int N_REF = (int)(sizeof(REF_TUNE) / sizeof(REF_TUNE[0]));

/* ---------- 状态 ---------- */
static int g_st_kc = 0, g_st_nc = 0, g_mt_kc = 0, g_mt_nc = 0;
static int g_have_cache = 0;          /* 1 = 已加载/生成有效调参 */
static int g_inited = 0;
static int g_tuning = 0;              /* 调参进行中, 阻止递归触发 */
static char g_status[256];

/* INT8 GEMM 调参 (与 float GEMM 同构: 运行时实测最优 KC/NC, 持久化复用)。
 * INT8 打包面板每元素仅 1 字节 (float 为 4 字节), 故同等 L2 容量下可容纳更大的 KC/NC,
 * 提升 A/B 面板复用; 本模块以更松的面板拟合约束 + 同款无回归门控落地。 */
static int g_i8_st_kc = 0, g_i8_st_nc = 0, g_i8_mt_kc = 0, g_i8_mt_nc = 0;
static int g_have_i8 = 0;             /* 1 = INT8 调参已就绪 (缓存或参考表) */

/* ---------- host 签名 ---------- */
static std::string host_signature(void) {
    std::string brand = "unknown";
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* p = strchr(line, ':');
                if (p) { brand = std::string(p + 1); break; }
            }
        }
        fclose(f);
    }
    while (!brand.empty() && (brand.back() == ' ' || brand.back() == '\n' || brand.back() == '\t'))
        brand.pop_back();
    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long l3 = sysconf(_SC_LEVEL3_CACHE_SIZE);
    char buf[512];
    snprintf(buf, sizeof(buf), "%s|L2=%ld|L3=%ld", brand.c_str(), l2, l3);
    /* 简单 FNV 风格哈希, 避免引入额外依赖 */
    unsigned long long h = 1469598103934665603ULL;
    for (const char* s = buf; *s; ++s) { h ^= (unsigned char)*s; h *= 1099511628211ULL; }
    char sig[64];
    snprintf(sig, sizeof(sig), "%016llx", (unsigned long long)(h & 0xffffffffffffffffULL));
    return std::string(sig);
}

static std::string cache_path(void) {
    const char* home = std::getenv("HOME");
    if (!home) home = "/tmp";
    char path[1024];
    snprintf(path, sizeof(path), "%s/.cache/acfun/acfun_tune_%s.txt", home, host_signature().c_str());
    return std::string(path);
}

static void ensure_cache_dir(const std::string& p) {
    std::string dir = p.substr(0, p.find_last_of('/'));
    mkdir(dir.c_str(), 0755);   /* 逐层创建较粗糙, 但 ~/.cache/acfun 仅一层 */
    /* 兜底: 若父目录不存在则再创建 */
    std::string parent = dir.substr(0, dir.find_last_of('/'));
    if (!parent.empty()) mkdir(parent.c_str(), 0755);
}

static bool load_cache(void) {
    std::string p = cache_path();
    FILE* f = fopen(p.c_str(), "r");
    if (!f) return false;
    char line[256];
    bool ok = false, ok_i8 = false;
    while (fgets(line, sizeof(line), f)) {
        int skc=0,snc=0,mkc=0,mnc=0;
        if (sscanf(line, "TUNE st_kc=%d st_nc=%d mt_kc=%d mt_nc=%d", &skc,&snc,&mkc,&mnc) == 4
            && skc>0 && snc>0 && mkc>0 && mnc>0) {
            g_st_kc=skc; g_st_nc=snc; g_mt_kc=mkc; g_mt_nc=mnc; ok=true;
        }
        int i8skc=0,i8snc=0,i8mkc=0,i8mnc=0;
        if (sscanf(line, "TUNE st_kc=%*d st_nc=%*d mt_kc=%*d mt_nc=%*d "
                        "i8_st_kc=%d i8_st_nc=%d i8_mt_kc=%d i8_mt_nc=%d",
                 &i8skc,&i8snc,&i8mkc,&i8mnc) == 4
            && i8skc>0 && i8snc>0 && i8mkc>0 && i8mnc>0) {
            g_i8_st_kc=i8skc; g_i8_st_nc=i8snc; g_i8_mt_kc=i8mkc; g_i8_mt_nc=i8mnc; ok_i8=true;
        }
    }
    fclose(f);
    g_have_cache = ok ? 1 : 0;
    g_have_i8 = ok_i8 ? 1 : 0;   /* 旧缓存无 i8 字段时回落参考表/启发式, 不阻断 float 调参 */
    return ok;
}

static void save_cache(void) {
    std::string p = cache_path();
    ensure_cache_dir(p);
    FILE* f = fopen(p.c_str(), "w");
    if (!f) return;
    fprintf(f, "TUNE st_kc=%d st_nc=%d mt_kc=%d mt_nc=%d", g_st_kc, g_st_nc, g_mt_kc, g_mt_nc);
    if (g_have_i8)
        fprintf(f, " i8_st_kc=%d i8_st_nc=%d i8_mt_kc=%d i8_mt_nc=%d",
                g_i8_st_kc, g_i8_st_nc, g_i8_mt_kc, g_i8_mt_nc);
    fprintf(f, "\n");
    fclose(f);
}

/* ---------- 单次调参: 在多尺寸套件上扫描 KC/NC 网格, 取稳健最优 ----------
 * 关键约束(避免回归): 候选 (KC,NC) 必须使单 MC 块 A 面板 (MCnom×KC) 与 NC 块 B 面板
 * (KC×NC) 合计稳居 L2 (<= L2/2), 否则大分块会撑爆 L2、在中小矩阵上反而变慢;
 * 且仅当某候选在套件全部尺寸上都不劣于启发式基线(容 3% 噪声)时才采纳, 否则回落启发式。 */
static void tune_mode(int nt, const int* sizes, int n_sizes, int* out_kc, int* out_nc) {
    /* 预分配最大尺寸缓冲, 各尺寸切片复用 */
    int Smax = 0;
    for (int i = 0; i < n_sizes; ++i) if (sizes[i] > Smax) Smax = sizes[i];
    std::vector<float> A((size_t)Smax*Smax), B((size_t)Smax*Smax), C((size_t)Smax*Smax);
    static std::mt19937 g(98765);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (size_t i = 0; i < A.size(); ++i) A[i] = d(g);
    for (size_t i = 0; i < B.size(); ++i) B[i] = d(g);

    acfun_set_threads(nt);
    acfun_set_num_threads_runtime(nt);

    /* 启发式基线耗时(各尺寸, env KC=256/NC=128, 即 L2 启发式在超大 L2 机器上的取值) */
    std::vector<double> base_ms(n_sizes, 1e30);
    {
        setenv("ACFUN_KC", "256", 1);
        setenv("ACFUN_NC", "128", 1);
        for (int s = 0; s < n_sizes; ++s) {
            int S = sizes[s];
            for (int w = 0; w < 1; ++w) acfun_gemm(S, S, S, A.data(), B.data(), C.data());
            double ms = 1e30;
            for (int r = 0; r < 2; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                acfun_gemm(S, S, S, A.data(), B.data(), C.data());
                auto t1 = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (dt < ms) ms = dt;
            }
            base_ms[s] = ms;
        }
        unsetenv("ACFUN_KC");
        unsetenv("ACFUN_NC");
    }

    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long l2fit = (l2 > 0) ? (l2 / 2) : (1L << 19);   /* 单面板稳居 L2 的容量上限 */
    const long MCnom = 256;

    int best_kc = 256, best_nc = 128;
    double best_gm = 1e30;
    char env_kc[32], env_nc[32];
    for (int i = 0; i < N_KC; ++i) {
        for (int j = 0; j < N_NC; ++j) {
            int kc = KC_LIST[i], nc = NC_LIST[j];
            /* L2 面板拟合约束: A 单 MC 块面板 + B 单 NC 块面板 <= L2/2 */
            if ((long)kc * nc * 4 + (long)MCnom * kc * 4 > l2fit) continue;
            snprintf(env_kc, sizeof(env_kc), "%d", kc);
            snprintf(env_nc, sizeof(env_nc), "%d", nc);
            setenv("ACFUN_KC", env_kc, 1);
            setenv("ACFUN_NC", env_nc, 1);
            double gm = 1.0;
            bool valid = true;
            for (int s = 0; s < n_sizes; ++s) {
                int S = sizes[s];
                for (int w = 0; w < 1; ++w) acfun_gemm(S, S, S, A.data(), B.data(), C.data());
                double ms = 1e30;
                for (int r = 0; r < 2; ++r) {
                    auto t0 = std::chrono::steady_clock::now();
                    acfun_gemm(S, S, S, A.data(), B.data(), C.data());
                    auto t1 = std::chrono::steady_clock::now();
                    double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (dt < ms) ms = dt;
                }
                /* 任一尺寸较启发式基线退化超 3% 则整体否决(防止中小矩阵回归) */
                if (ms > base_ms[s] * 1.03) { valid = false; break; }
                gm *= ms;
            }
            unsetenv("ACFUN_KC");
            unsetenv("ACFUN_NC");
            if (!valid) continue;
            gm = std::pow(gm, 1.0 / n_sizes);   /* 几何平均(套件整体提速) */
            if (gm < best_gm) { best_gm = gm; best_kc = kc; best_nc = nc; }
        }
    }
    *out_kc = best_kc; *out_nc = best_nc;
}

/* INT8 GEMM 调参: 与 tune_mode 同构, 但作用在 uint8×int8 GEMM (VNNI) 上, 且使用随机 int8 输入。
 * 打包面板因 4×4×4 冗余布局同样约 4 字节/K-元素, 故沿用同一套 L2 面板拟合约束与无回归门控。 */
static void tune_mode_int8(int nt, const int* sizes, int n_sizes, int* out_kc, int* out_nc) {
    int Smax = 0;
    for (int i = 0; i < n_sizes; ++i) if (sizes[i] > Smax) Smax = sizes[i];
    std::vector<uint8_t> A((size_t)Smax*Smax);
    std::vector<int8_t>  B((size_t)Smax*Smax);
    std::vector<int32_t> C((size_t)Smax*Smax);
    static std::mt19937 g(13579);
    std::uniform_int_distribution<int> da(0, 255), db(-128, 127);
    for (size_t i = 0; i < A.size(); ++i) A[i] = (uint8_t)da(g);
    for (size_t i = 0; i < B.size(); ++i) B[i] = (int8_t)db(g);

    acfun_set_threads(nt);
    acfun_set_num_threads_runtime(nt);

    std::vector<double> base_ms(n_sizes, 1e30);
    {
        setenv("ACFUN_KC", "256", 1);
        setenv("ACFUN_NC", "256", 1);
        for (int s = 0; s < n_sizes; ++s) {
            int S = sizes[s];
            for (int w = 0; w < 1; ++w) acfun_gemm_int8(S, S, S, A.data(), B.data(), C.data());
            double ms = 1e30;
            for (int r = 0; r < 2; ++r) {
                auto t0 = std::chrono::steady_clock::now();
                acfun_gemm_int8(S, S, S, A.data(), B.data(), C.data());
                auto t1 = std::chrono::steady_clock::now();
                double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                if (dt < ms) ms = dt;
            }
            base_ms[s] = ms;
        }
        unsetenv("ACFUN_KC");
        unsetenv("ACFUN_NC");
    }

    long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
    long l2fit = (l2 > 0) ? (l2 / 2) : (1L << 19);
    const long MCnom = 256;

    int best_kc = 256, best_nc = 256;
    double best_gm = 1e30;
    char env_kc[32], env_nc[32];
    for (int i = 0; i < N_KC; ++i) {
        for (int j = 0; j < N_NC; ++j) {
            int kc = KC_LIST[i], nc = NC_LIST[j];
            if ((long)kc * nc * 4 + (long)MCnom * kc * 4 > l2fit) continue;
            snprintf(env_kc, sizeof(env_kc), "%d", kc);
            snprintf(env_nc, sizeof(env_nc), "%d", nc);
            setenv("ACFUN_KC", env_kc, 1);
            setenv("ACFUN_NC", env_nc, 1);
            double gm = 1.0;
            bool valid = true;
            for (int s = 0; s < n_sizes; ++s) {
                int S = sizes[s];
                for (int w = 0; w < 1; ++w) acfun_gemm_int8(S, S, S, A.data(), B.data(), C.data());
                double ms = 1e30;
                for (int r = 0; r < 2; ++r) {
                    auto t0 = std::chrono::steady_clock::now();
                    acfun_gemm_int8(S, S, S, A.data(), B.data(), C.data());
                    auto t1 = std::chrono::steady_clock::now();
                    double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
                    if (dt < ms) ms = dt;
                }
                if (ms > base_ms[s] * 1.03) { valid = false; break; }
                gm *= ms;
            }
            unsetenv("ACFUN_KC");
            unsetenv("ACFUN_NC");
            if (!valid) continue;
            gm = std::pow(gm, 1.0 / n_sizes);
            if (gm < best_gm) { best_gm = gm; best_kc = kc; best_nc = nc; }
        }
    }
    *out_kc = best_kc; *out_nc = best_nc;
}

static void do_tune(void) {
    g_tuning = 1;   /* 阻止 acfun_get_tuned_kc_nc 在调参过程中再次触发调参(递归) */
    int st_kc, st_nc, mt_kc, mt_nc;
    /* 单线程套件: 中小矩阵 + 典型卷积/FC 尺寸; 多线程套件: 覆盖小→大以兼顾负载均衡 */
    int st_sizes[] = {256, 512, 1024};
    int mt_sizes[] = {512, 1024, 2048, 4096};
    tune_mode(1, st_sizes, (int)(sizeof(st_sizes)/sizeof(st_sizes[0])), &st_kc, &st_nc);
    int nt = acfun_effective_cores();
    if (nt < 1) nt = 1;
    tune_mode(nt, mt_sizes, (int)(sizeof(mt_sizes)/sizeof(mt_sizes[0])), &mt_kc, &mt_nc);
    g_st_kc = st_kc; g_st_nc = st_nc; g_mt_kc = mt_kc; g_mt_nc = mt_nc;
    g_have_cache = 1;

    /* INT8 GEMM 调参 (v1.0.0 新增): 同机实测最优 KC/NC, 与 float 调参同缓存持久化 */
    int i8_st_kc, i8_st_nc, i8_mt_kc, i8_mt_nc;
    int i8_st_sizes[] = {256, 512, 1024};
    int i8_mt_sizes[] = {512, 1024, 2048, 4096};
    tune_mode_int8(1, i8_st_sizes, (int)(sizeof(i8_st_sizes)/sizeof(i8_st_sizes[0])), &i8_st_kc, &i8_st_nc);
    tune_mode_int8(nt, i8_mt_sizes, (int)(sizeof(i8_mt_sizes)/sizeof(i8_mt_sizes[0])), &i8_mt_kc, &i8_mt_nc);
    g_i8_st_kc = i8_st_kc; g_i8_st_nc = i8_st_nc; g_i8_mt_kc = i8_mt_kc; g_i8_mt_nc = i8_mt_nc;
    g_have_i8 = 1;

    save_cache();
    g_tuning = 0;
    snprintf(g_status, sizeof(g_status),
             "tuned(ST KC=%d NC=%d / MT KC=%d NC=%d, i8 MT NC=%d, cached)",
             st_kc, st_nc, mt_kc, mt_nc, i8_mt_nc);
    printf("[acfun] 自动调参完成 (ST: KC=%d NC=%d, MT: KC=%d NC=%d, INT8 MT NC=%d) -> %s\n",
           st_kc, st_nc, mt_kc, mt_nc, i8_mt_nc, cache_path().c_str());
}

/* 惰性初始化: 仅首次查询时决定走"强制调参 / 读缓存 / 参考表 / 启发式" */
static void ensure_init(void) {
    if (g_inited) return;
    g_inited = 1;
    if (!g_tuning && std::getenv("ACFUN_AUTOTUNE")) {
        do_tune();
        snprintf(g_status, sizeof(g_status),
                 "tuned(ST KC=%d NC=%d / MT KC=%d NC=%d, cached)",
                 g_st_kc, g_st_nc, g_mt_kc, g_mt_nc);
        return;
    }
    if (load_cache()) {
        if (g_have_i8)
            snprintf(g_status, sizeof(g_status),
                     "cache(ST KC=%d NC=%d / MT KC=%d NC=%d, i8 MT NC=%d)",
                     g_st_kc, g_st_nc, g_mt_kc, g_mt_nc, g_i8_mt_nc);
        else
            snprintf(g_status, sizeof(g_status),
                     "cache(ST KC=%d NC=%d / MT KC=%d NC=%d)",
                     g_st_kc, g_st_nc, g_mt_kc, g_mt_nc);
        return;
    }
    snprintf(g_status, sizeof(g_status), "heuristic(L2)");
}

int acfun_get_tuned_kc_nc(int nt, int* KC, int* NC) {
    ensure_init();
    if (g_have_cache) {
        if (nt > 1) { *KC = g_mt_kc; *NC = g_mt_nc; }
        else        { *KC = g_st_kc; *NC = g_st_nc; }
        return 1;
    }
    /* 未调参: 查参考表(按 CPU 型号) */
    std::string brand = "unknown";
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* p = strchr(line, ':');
                if (p) { brand = std::string(p + 1); break; }
            }
        }
        fclose(f);
    }
    for (int i = 0; i < N_REF; ++i) {
        if (brand.find(REF_TUNE[i].key) != std::string::npos) {
            if (nt > 1) { *KC = REF_TUNE[i].mt_kc; *NC = REF_TUNE[i].mt_nc; }
            else        { *KC = REF_TUNE[i].st_kc; *NC = REF_TUNE[i].st_nc; }
            return 1;
        }
    }
    return 0;   /* 回落静态启发式 */
}

/* INT8 GEMM 调参查询: 返回 1 表示取到调参值(缓存或参考表), 0 表示回落启发式。
 * int8 与 float 共享同一套 host 签名与候选网格, 仅面板拟合约束与无回归门控复用。 */
int acfun_get_tuned_kc_nc_int8(int nt, int* KC, int* NC) {
    ensure_init();
    if (g_have_i8) {
        if (nt > 1) { *KC = g_i8_mt_kc; *NC = g_i8_mt_nc; }
        else        { *KC = g_i8_st_kc; *NC = g_i8_st_nc; }
        return 1;
    }
    std::string brand = "unknown";
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* p = strchr(line, ':');
                if (p) { brand = std::string(p + 1); break; }
            }
        }
        fclose(f);
    }
    for (int i = 0; i < N_REF; ++i) {
        if (brand.find(REF_TUNE[i].key) != std::string::npos) {
            if (nt > 1) { *KC = REF_TUNE[i].mt_kc; *NC = REF_TUNE[i].mt_nc; }
            else        { *KC = REF_TUNE[i].st_kc; *NC = REF_TUNE[i].st_nc; }
            return 1;
        }
    }
    return 0;
}

void acfun_autotune(void) {
    g_inited = 0;   /* 强制重新初始化(即使之前已查过启发式) */
    g_have_cache = 0;
    do_tune();
}

const char* acfun_tune_status(void) {
    ensure_init();
    return g_status;
}
