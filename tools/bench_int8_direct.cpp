/* bench_int8_direct.cpp — 小 K INT8 直接卷积内核微基准 (v2.0.0)。
 *
 * 目的: 对 acfun_conv2d_int8_direct 内核做多轮计时取中位数, 消除共享主机墙钟
 * 噪声, 用于 A/B 对比 tap 解码消除前后 (v1.9.0 vs v2.0.0) 的稳定加速比。
 * 只测内核本身 (输入/权重预先量化), 新旧差异纯粹来自内核实现。
 *
 * 编译 (仓库根):
 *   g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
 *       tools/bench_int8_direct.cpp build/libacfun.a -fopenmp -o /tmp/bench_int8_direct
 *   /tmp/bench_int8_direct            # 1T 与 4T, 每场景 30 轮取中位数
 */
#include "acfun.h"
#include "internal.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstring>

using Clock = std::chrono::steady_clock;
static unsigned rng = 12345;
static float frand() { rng = rng * 1103515245u + 12345u; return ((rng >> 8) & 0xffff) / (float)0xffff - 0.5f; }

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Case { int H, W, Cin, Cout, KH, KW, st, pd; const char* name; };

static void bench_one(const Case& c, int threads) {
    int N = 1;
    int OH = (c.H + 2 * c.pd - c.KH) / c.st + 1;
    int OW = (c.W + 2 * c.pd - c.KW) / c.st + 1;
    int K = c.KH * c.KW * c.Cin;
    size_t inN = (size_t)N * c.H * c.W * c.Cin, wN = (size_t)K * c.Cout;
    size_t outN = (size_t)N * OH * OW * c.Cout;
    std::vector<float> in(inN), w(wN), b(c.Cout, 0.1f);
    for (auto& x : in) x = frand();
    for (auto& x : w) x = frand();
    float maxa = 0; for (float x : in) maxa = std::max(maxa, std::fabs(x));
    float inv = 127.0f / maxa, sa = maxa / 127.0f, ZP = 128;
    std::vector<uint8_t> inQ(inN);
    std::vector<int8_t> wq(wN);
    std::vector<float> ws(c.Cout);
    acfun_quantize_u8(in.data(), inQ.data(), inN, inv, (int)ZP);
    acfun_quantize_weights(w.data(), K, c.Cout, ws.data(), wq.data());
    std::vector<float> alpha(c.Cout), beta(c.Cout);
    for (int j = 0; j < c.Cout; ++j) {
        float Sw = 0; for (int k = 0; k < K; ++k) Sw += wq[(size_t)k * c.Cout + j];
        alpha[j] = sa * ws[j];
        beta[j]  = b[j] - alpha[j] * ZP * Sw;
    }
    std::vector<float> out(outN);

    acfun_set_num_threads_runtime(threads);
    for (int r = 0; r < 3; ++r)
        acfun_conv2d_int8_direct(inQ.data(), N, c.H, c.W, c.Cin, wq.data(), alpha.data(),
                                 beta.data(), c.KH, c.KW, c.Cout, c.st, c.pd, OH, OW, out.data(), 0);

    const int rounds = 30;
    std::vector<double> times;
    times.reserve(rounds);
    volatile float sink = 0;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = Clock::now();
        acfun_conv2d_int8_direct(inQ.data(), N, c.H, c.W, c.Cin, wq.data(), alpha.data(),
                                 beta.data(), c.KH, c.KW, c.Cout, c.st, c.pd, OH, OW, out.data(), 0);
        auto t1 = Clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        sink += out[r % outN];
    }
    acfun_set_num_threads_runtime(1);
    double med = median(times);
    double lo = *std::min_element(times.begin(), times.end());
    printf("  %-26s %dT  median=%8.3fms  min=%8.3fms  (chk=%.3f)\n",
           c.name, threads, med, lo, (double)sink);
}

int main() {
    /* 仅列直接内核设计范围内的场景 (路由条件 K<=64 且非 zero-copy 1x1)。
     * 112x112x16 k3 (K=144) 走 Winograd, 不经本内核, 故不列入。 */
    Case cases[] = {
        {224, 224, 3, 16, 3, 3, 1, 1, "224x224x3->16 k3 s1 p1"},
        {56,  56, 16, 32, 1, 1, 2, 0, "56x56x16->32 k1 s2 p0"},
        {20,  20, 3, 13, 3, 3, 2, 1, "20x20x3->13 k3 s2 p1"},
    };
    printf("小 K INT8 直接卷积内核微基准 (30 轮取中位数)\n");
    for (auto& c : cases) {
        bench_one(c, 1);
        bench_one(c, 4);
    }
    return 0;
}
