/* Standalone isolation test: INT8 Winograd (3x3) vs float conv correctness, 多 N 验证 batch 偏移.
 * Build: g++ -O3 -mavx512f -mavx512bw -mavx512dq -mfma tools/test_int8_iso.cpp
 *       -Iinclude -Iinclude/acfun -Isrc build/libacfun.a -fopenmp -o /tmp/ti */
#include "acfun.h"
#include "internal.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

static void rand_fill(float* p, size_t n, float s) {
    static std::mt19937 g(12345);
    std::uniform_real_distribution<float> d(-s, s);
    for (size_t i = 0; i < n; ++i) p[i] = d(g);
}

int main() {
    int H=112,W=112,Cin=16,Cout=16,KH=3,KW=3,st=1,pd=1;
    std::vector<float> w((size_t)KH*KW*Cin*Cout), b(Cout,0.1f);
    rand_fill(w.data(), w.size(), 1.0f);
    std::vector<float> ws(Cout); std::vector<int8_t> wq((size_t)KH*KW*Cin*Cout);
    acfun_quantize_weights(w.data(),KH*KW*Cin,Cout,ws.data(),wq.data());
    for (int N : {1, 2, 3}) {
        int OH=H,OW=W;
        size_t inN=(size_t)N*H*W*Cin, outN=(size_t)N*OH*OW*Cout;
        std::vector<float> in(inN); rand_fill(in.data(), inN, 1.0f);
        std::vector<float> of(outN), oi(outN);
        acfun_conv2d(in.data(),N,H,W,Cin,w.data(),b.data(),KH,KW,Cout,st,pd,of.data());
        float maxa=0; for(size_t i=0;i<inN;++i) maxa=std::max(maxa,std::fabs(in[i]));
        float inv_s=127.0f/maxa, sa=maxa/127.0f;
        acfun_conv2d_int8(in.data(),N,H,W,Cin,wq.data(),ws.data(),b.data(),KH,KW,Cout,st,pd,sa,oi.data(),0);
        float maxd=0, norm=0;
        for(size_t i=0;i<outN;++i){ maxd=std::max(maxd,std::fabs(of[i]-oi[i])); norm=std::max(norm,std::fabs(of[i])); }
        printf("N=%d  INT8 Winograd via conv2d_int8: max|Δ|=%.3e  rel=%.2f%% %s\n",
               N, maxd, 100.0f*maxd/(norm+1e-9f), maxd<0.5f?"OK":"FAIL");
    }
    return 0;
}
