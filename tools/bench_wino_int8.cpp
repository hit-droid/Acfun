/* bench_wino_int8.cpp — INT8 Winograd F(2,3) 输入变换优化隔离验证 + 稳定 A/B 微基准。
 *
 * 直接调用内部 API acfun_winograd_conv3x3_int8(输入为 float, 内部就地量化), 避开
 * conv_int8 路由与量化遍干扰, 只测 Winograd 内核本身。
 *
 * 用法 (仓库根):
 *   g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
 *       tools/bench_wino_int8.cpp build/libacfun.a -fopenmp -o /tmp/bench_wino_int8
 *   /tmp/bench_wino_int8
 *
 * 输出:
 *   - 每个配置输出的 64-bit FNV-1a hash (对 float bit 模式) —— 两版 hash 逐位相同即证明
 *     数值严格等价 (max|Δ|=0); 覆盖奇数尺寸/非16倍数通道/padding 边界。
 *   - 主攻场景 30 轮中位数 + 最小值, 用于稳定加速比 A/B。
 */
#include "acfun.h"
#include "internal.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>

using Clock = std::chrono::steady_clock;
static uint32_t rng = 0x9e3779b9u;
static float frand() { rng = rng * 1664525u + 1013904223u; return ((rng >> 8) & 0xffff) / (float)0xffff - 0.5f; }

static uint64_t hash_out(const float* p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) {
        uint32_t b; std::memcpy(&b, &p[i], 4);
        for (int j = 0; j < 4; ++j) { h ^= (b >> (j * 8)) & 0xff; h *= 1099511628211ULL; }
    }
    return h;
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

struct Case { int H, W, Cin, Cout; bool timed; };

static void run_one(const Case& c) {
    const int KH = 3, KW = 3, st = 1, pd = 1, N = 1;
    const int OH = c.H, OW = c.W;
    const size_t inN = (size_t)N * c.H * c.W * c.Cin;
    const size_t wN  = (size_t)KH * KW * c.Cin * c.Cout;
    const size_t outN = (size_t)N * OH * OW * c.Cout;
    std::vector<float> in(inN), w(wN), b(c.Cout, 0.1f), out(outN);
    for (auto& x : in) x = frand();
    for (auto& x : w)  x = frand();
    std::vector<float> ws(c.Cout); std::vector<int8_t> wq(wN);
    acfun_quantize_weights(w.data(), KH * KW * c.Cin, c.Cout, ws.data(), wq.data());
    float maxa = 0; for (float x : in) maxa = std::max(maxa, std::fabs(x));
    const float inv = 127.0f / maxa, s_in = maxa / 127.0f;

    /* 预热 (JIT/缓存/线程池稳定) */
    for (int r = 0; r < 3; ++r)
        acfun_winograd_conv3x3_int8(in.data(), N, c.H, c.W, c.Cin, wq.data(), ws.data(),
                                    b.data(), c.Cout, s_in, inv, out.data(), 0);
    uint64_t h = hash_out(out.data(), outN);

    if (!c.timed) {
        printf("  OK  %3dx%3dx%2d->%-2d  hash=%016llx\n", c.H, c.W, c.Cin, c.Cout,
               (unsigned long long)h);
        return;
    }
    const int rounds = 30;
    std::vector<double> t; t.reserve(rounds);
    volatile float sink = 0;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = Clock::now();
        acfun_winograd_conv3x3_int8(in.data(), N, c.H, c.W, c.Cin, wq.data(), ws.data(),
                                    b.data(), c.Cout, s_in, inv, out.data(), 0);
        auto t1 = Clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        sink += out[r % outN];
    }
    printf("  TM   %3dx%3dx%2d->%-2d  median=%8.3fms  min=%8.3fms  hash=%016llx (chk=%.2f)\n",
           c.H, c.W, c.Cin, c.Cout, median(t), *std::min_element(t.begin(), t.end()),
           (unsigned long long)h, (double)sink);
}

int main() {
    printf("INT8 Winograd 输入变换优化 — 正确性与稳定 A/B (30 轮中位数)\n");
    printf("[正确性: 两版 hash 必须逐位相同]\n");
    Case corr[] = {
        {7, 7, 4, 4, false}, {9, 7, 8, 16, false}, {16, 16, 16, 16, false},
        {15, 13, 20, 13, false}, {17, 17, 24, 40, false}, {8, 8, 16, 64, false},
        {5, 5, 32, 8, false}, {13, 11, 12, 20, false}, {32, 28, 48, 33, false},
    };
    for (auto& c : corr) run_one(c);
    printf("[计时: 主攻小通道场景]\n");
    Case timed[] = {
        {112, 112, 16, 16, true}, {113, 97, 16, 48, true}, {24, 24, 8, 64, true},
        {56, 56, 32, 32, true}, {28, 28, 64, 64, true}, {64, 64, 32, 64, true},
    };
    for (auto& c : timed) run_one(c);
    return 0;
}
