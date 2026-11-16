/* 独立小通道组分组卷积验证: fused 直接内核 vs 朴素分组参考。
 * 覆盖宽路由条件 (Cin_per≤8, Cout_per≤16) 下的 stride/pad/非方阵/奇偶边界,
 * 以及有/无偏置、有/无 ReLU 的全部组合。用法:
 *   g++ -O3 -std=c++17 -fopenmp -Iinclude -Iinclude/acfun \
 *       tools/test_group_conv.cpp build/libacfun.a -o /tmp/test_group_conv && /tmp/test_group_conv */
#include "acfun.h"
#include "internal.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

static void rf(std::vector<float>& v, unsigned seed) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& x : v) x = d(g);
}

/* 朴素分组参考: 权重 [g][kh][kw][ic][oc], 输出 NHWC (与 benchmark 内参考同构) */
static void naive_grouped(const float* in, int N, int H, int W, int Cin,
                          const float* w, const float* b, int KH, int KW, int Cout,
                          int st, int pd, int G, float* out, int act) {
    int Cp = Cin / G, Cop = Cout / G;
    int OH = (H + 2 * pd - KH) / st + 1, OW = (W + 2 * pd - KW) / st + 1;
    for (int g = 0; g < G; ++g)
        for (int n = 0; n < N; ++n)
            for (int oc = 0; oc < Cop; ++oc) {
                const float* wg = w + (size_t)g * Cop * KH * KW * Cp;
                const float* bg = b ? b + g * Cop : nullptr;
                for (int oh = 0; oh < OH; ++oh)
                    for (int ow = 0; ow < OW; ++ow) {
                        float s = bg ? bg[oc] : 0.0f;
                        for (int kh = 0; kh < KH; ++kh) {
                            int ih = oh * st - pd + kh; if (ih < 0 || ih >= H) continue;
                            for (int kw = 0; kw < KW; ++kw) {
                                int iw = ow * st - pd + kw; if (iw < 0 || iw >= W) continue;
                                for (int ic = 0; ic < Cp; ++ic) {
                                    size_t idx = ((size_t)n * H + ih) * W + iw;
                                    size_t ia = (size_t)g * Cp + idx * Cin + ic;
                                    s += in[ia] * wg[((kh * KW + kw) * Cp + ic) * Cop + oc];
                                }
                            }
                        }
                        if (act == 1 && s < 0.0f) s = 0.0f;
                        size_t odx = ((size_t)n * OH + oh) * OW + ow;
                        out[odx * (size_t)Cout + (size_t)g * Cop + oc] = s;
                    }
            }
}

int main() {
    struct Cfg { int H, W, Cin, Cout, KH, KW, st, pd, G; };
    std::vector<Cfg> cfgs = {
        {56,56,16,32,3,3,1,1,4},   // 基准已有 (fused)
        {28,28,32,64,1,1,1,0,8},   // 基准已有 1x1 (fused)
        {14,14,24,48,3,3,2,1,6},   // 基准已有 stride2 (fused)
        { 9,11,12,24,3,3,1,1,4},   // 奇数 / 非整除
        {16,16, 8,32,3,3,1,1,8},   // Cin_per=1, Cout_per=4
        {16,16,16,64,3,1,1,0,8},   // 非方阵 3x1
        {20,18,32,32,5,5,1,2,8},   // 5x5 / pad2
        {13,13,24,48,3,3,2,0,6},   // pad0 / 奇数
        {32,32,64,64,1,1,2,1,8},   // 1x1 s2 p1, 边界 Cp=Cop=8
        { 7, 9,16,16,3,3,1,1,4},   // 小图
        {40,40,48,96,3,3,1,1,12},  // g12
        {24,24,16,128,1,1,1,0,4},  // Cout_per=32 (路由外, 原路径)
        {56,56,32,32,3,3,1,1,32},  // 深度卷积 (DW 路径)
    };
    printf("=== 小通道组分组卷积正确性 (fused vs 朴素, 每配置 4 种 bias/act 组合) ===\n");
    int fails = 0;
    for (size_t ci = 0; ci < cfgs.size(); ++ci) {
        auto c = cfgs[ci];
        int OH = (c.H + 2 * c.pd - c.KH) / c.st + 1;
        int OW = (c.W + 2 * c.pd - c.KW) / c.st + 1;
        size_t inN = (size_t)c.H * c.W * c.Cin;
        size_t wN  = (size_t)c.G * (c.Cout / c.G) * c.KH * c.KW * (c.Cin / c.G);
        size_t outN = (size_t)OH * OW * c.Cout;
        float worst = 0.0f;
        bool all_ok = true;
        for (int wb = 0; wb < 2; ++wb)
            for (int act = 0; act < 2; ++act) {
                std::vector<float> in(inN), w(wN), b(c.Cout, 0.0f), ore(outN), oa(outN);
                rf(in, (unsigned)(ci * 10 + 1));
                rf(w,  (unsigned)(ci * 10 + 2));
                if (wb) rf(b, (unsigned)(ci * 10 + 3));
                const float* bp = wb ? b.data() : nullptr;
                naive_grouped(in.data(), 1, c.H, c.W, c.Cin, w.data(), bp, c.KH, c.KW,
                              c.Cout, c.st, c.pd, c.G, ore.data(), act);
                acfun_conv2d_g(in.data(), 1, c.H, c.W, c.Cin, w.data(), bp, c.KH, c.KW,
                               c.Cout, c.st, c.pd, c.G, oa.data(), act);
                float md = 0.0f;
                for (size_t i = 0; i < outN; ++i) md = std::max(md, std::fabs(ore[i] - oa[i]));
                worst = std::max(worst, md);
                if (!(md < 1e-2f)) all_ok = false;
            }
        if (!all_ok) ++fails;
        printf("  %2dx%2dx%2d->%3d k%d g%-2d s%d p%d  最大 max|Δ|=%9.3e  %s\n",
               c.H, c.W, c.Cin, c.Cout, c.KH, c.G, c.st, c.pd, worst, all_ok ? "OK" : "FAIL");
    }
    printf("  正确性: %s\n", fails ? "存在失败" : "全部通过");
    return fails ? 1 : 0;
}
