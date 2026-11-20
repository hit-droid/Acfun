/* test_int8_direct.cpp — 小 K INT8 直接卷积 (v1.7.0) 隔离正确性回归测试。
 *
 * 验证 acfun_conv2d_int8_direct (AVX-512 VNNI 内核) 与原 im2col+GEMM epilogue
 * 路径逐元素严格等价 (max|Δ|=0)。覆盖 1x1 / 3x3 padding / stride>1 / 非整除
 * Cout / batch>1 / 有无 ReLU。任何直接内核的数值改动都应重跑本测试。
 *
 * 编译 (仓库根):
 *   cmake -S . -B build && cmake --build build -j
 *   g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
 *       tools/test_int8_direct.cpp build/libacfun.a -fopenmp -o /tmp/test_int8_direct
 *   /tmp/test_int8_direct     # 期望 ALL PASS
 */
#include "acfun.h"
#include "internal.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstring>

static void ref_gemm_epi(int M, int N, int K, const uint8_t* A, const int8_t* B,
                         const float* alpha, const float* beta, float* C, int act) {
    for (int i = 0; i < M; ++i) {
        const uint8_t* a = A + (size_t)i * K;
        for (int j = 0; j < N; ++j) {
            int acc = 0;
            for (int k = 0; k < K; ++k) acc += (int)a[k] * (int)B[(size_t)k * N + j];
            float v = (float)acc * alpha[j] + beta[j];
            if (act && v < 0) v = 0;
            C[i * N + j] = v;
        }
    }
}

static void im2col_u8(const uint8_t* in, int H, int W, int Cin,
                      int KH, int KW, int stride, int pad, int OH, int OW, uint8_t* col) {
    int ic = KH * KW * Cin;
    std::memset(col, 128, (size_t)OH * OW * ic);   /* padding 填 zp=128 */
    for (int oh = 0; oh < OH; ++oh) for (int ow = 0; ow < OW; ++ow) {
        uint8_t* dst = col + ((size_t)oh * OW + ow) * ic;
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh * stride - pad + kh;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow * stride - pad + kw;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                    std::memcpy(dst, in + ((size_t)ih * W + iw) * Cin, Cin);
                dst += Cin;
            }
        }
    }
}

static unsigned rng = 12345;
static float frand() { rng = rng * 1103515245u + 12345u; return ((rng >> 8) & 0xffff) / (float)0xffff - 0.5f; }

static bool one(int N, int H, int W, int Cin, int Cout, int KH, int KW, int st, int pd, int act) {
    int OH = (H + 2 * pd - KH) / st + 1, OW = (W + 2 * pd - KW) / st + 1;
    int K = KH * KW * Cin;
    size_t inN = (size_t)N * H * W * Cin, wN = (size_t)K * Cout, outN = (size_t)N * OH * OW * Cout;
    std::vector<float> in(inN), w(wN), b(Cout);
    for (auto& x : in) x = frand();
    for (auto& x : w) x = frand();
    for (auto& x : b) x = frand() * 0.1f;
    float maxa = 0; for (float x : in) maxa = std::max(maxa, std::fabs(x));
    float inv = 127.0f / maxa, sa = maxa / 127.0f, ZP = 128;
    std::vector<uint8_t> inQ(inN);
    std::vector<int8_t> wq(wN);
    std::vector<float> ws(Cout);
    acfun_quantize_u8(in.data(), inQ.data(), inN, inv, (int)ZP);
    acfun_quantize_weights(w.data(), K, Cout, ws.data(), wq.data());
    std::vector<float> alpha(Cout), beta(Cout);
    for (int j = 0; j < Cout; ++j) {
        float Sw = 0; for (int k = 0; k < K; ++k) Sw += wq[(size_t)k * Cout + j];
        alpha[j] = sa * ws[j];
        beta[j]  = b[j] - alpha[j] * ZP * Sw;
    }

    /* 参考: im2col(量化输入) + 严格标量 GEMM epilogue */
    std::vector<uint8_t> A((size_t)N * OH * OW * K);
    for (int n = 0; n < N; ++n)
        im2col_u8(inQ.data() + (size_t)n * H * W * Cin, H, W, Cin, KH, KW, st, pd, OH, OW,
                  A.data() + (size_t)n * OH * OW * K);
    std::vector<float> ref(outN), got(outN);
    ref_gemm_epi((int)(N * OH * OW), Cout, K, A.data(), wq.data(),
                 alpha.data(), beta.data(), ref.data(), act);

    /* 新直接内核 (单线程) */
    acfun_set_num_threads_runtime(1);
    acfun_conv2d_int8_direct(inQ.data(), N, H, W, Cin, wq.data(), alpha.data(), beta.data(),
                             KH, KW, Cout, st, pd, OH, OW, got.data(), act);
    float md = 0; for (size_t i = 0; i < outN; ++i) md = std::max(md, std::fabs(ref[i] - got[i]));

    /* v1.9.0: 多线程 (4T) 逐元素一致性 —— 与单线程严格 max|Δ|=0 (分割输出行, 无写冲突)。
     * 同时对 ref (im2col+标量 GEMM) 也必须一致。 */
    std::vector<float> gotMT(outN);
    acfun_set_num_threads_runtime(4);
    acfun_conv2d_int8_direct(inQ.data(), N, H, W, Cin, wq.data(), alpha.data(), beta.data(),
                             KH, KW, Cout, st, pd, OH, OW, gotMT.data(), act);
    acfun_set_num_threads_runtime(1);
    float mdMT = 0, mdMTref = 0;
    for (size_t i = 0; i < outN; ++i) {
        mdMT = std::max(mdMT, std::fabs(got[i] - gotMT[i]));
        mdMTref = std::max(mdMTref, std::fabs(ref[i] - gotMT[i]));
    }
    bool ok = (md < 1e-5f) && (mdMT == 0.0f) && (mdMTref < 1e-5f);
    printf("N%d %dx%dx%d->%d k%d s%d p%d act%d K=%d  max|Δ|=%10.3e  MT:1T=%9.3e ref=%9.3e  %s\n",
           N, H, W, Cin, Cout, KH, st, pd, act, K, md, mdMT, mdMTref, ok ? "OK" : "FAIL");
    return ok;
}

int main() {
    bool ok = true;
    ok &= one(1, 28, 28, 64, 64, 1, 1, 1, 0, 0);   /* 1x1 (内核层, 路由会绕开但内核须正确) */
    ok &= one(1, 28, 28, 64, 64, 1, 1, 1, 0, 1);
    ok &= one(1, 56, 56, 16, 32, 1, 1, 2, 0, 0);   /* 1x1 stride2 */
    ok &= one(1, 224, 224, 3, 16, 3, 3, 1, 1, 0);  /* 224²×3 极小 Cin + padding */
    ok &= one(1, 224, 224, 3, 16, 3, 3, 1, 1, 1);
    ok &= one(2, 13, 11, 7, 19, 3, 3, 1, 1, 0);    /* 非整除 Cout, 奇数尺寸, batch=2 */
    ok &= one(1, 9, 9, 5, 23, 1, 1, 1, 0, 1);
    ok &= one(2, 20, 20, 3, 13, 3, 3, 2, 1, 1);    /* stride2 + padding + ReLU */
    printf("\n%s\n", ok ? "ALL PASS" : "SOME FAIL");
    return ok ? 0 : 1;
}
