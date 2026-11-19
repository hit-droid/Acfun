/* 隔离回归 + 性能: 1x1 INT8 专用内核 (v1.8.0)
 * 参考 = acfun_gemm_int8_epi (通用 VNNI GEMM, 与 conv_int8.cpp 原路径一致);
 * 被测 = acfun_conv1x1_int8 (专用内核)。输入 inQ/权重 wq/alpha/beta 完全相同,
 * 二者须逐元素 max|Δ|=0。覆盖: 多种 M/K/Cout、非整除 Cout、有/无 ReLU、多线程。
 * 编译: g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc tools/test_conv1x1.cpp \
 *          build/libacfun.a -fopenmp -o /tmp/test_conv1x1  */
#include "acfun.h"
#include "internal.h"
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <algorithm>

static double now(){ return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
static uint8_t rnd8(unsigned s){ return (uint8_t)(s*2654435761u >> 24); }
static int8_t rndi8(unsigned s){ return (int8_t)((s*1103515245u >> 24) - 128); }

static bool run(int M, int K, int Cout, int act, int nt) {
    std::vector<uint8_t> A((size_t)M*K);
    std::vector<int8_t>  B((size_t)K*Cout);
    std::vector<float> alpha(Cout), beta(Cout), cref((size_t)M*Cout), cnew((size_t)M*Cout);
    unsigned s = (unsigned)(M*131 + K*17 + Cout);
    for (size_t i=0;i<A.size();++i) A[i]=rnd8(s++);
    for (size_t i=0;i<B.size();++i) B[i]=rndi8(s++);
    for (int j=0;j<Cout;++j){ alpha[j]=0.001f+0.0001f*(j%13); beta[j]=-0.5f+0.01f*(j%7); }

    acfun_set_num_threads_runtime(nt);
    acfun_gemm_int8_epi(M,Cout,K,A.data(),B.data(),cref.data(),alpha.data(),beta.data(),act);
    acfun_conv1x1_int8(A.data(),M,K,Cout,B.data(),alpha.data(),beta.data(),cnew.data(),act);

    float md=0; for(size_t i=0;i<cref.size();++i) md=std::max(md,std::fabs(cref[i]-cnew[i]));
    bool ok = (md==0.0f);

    /* 性能: 各取多次均值 */
    int reps = (M*Cout > (1u<<20)) ? 20 : 100;
    double t0=now(); for(int r=0;r<reps;++r) acfun_gemm_int8_epi(M,Cout,K,A.data(),B.data(),cref.data(),alpha.data(),beta.data(),act); double tg=(now()-t0)/reps;
    t0=now(); for(int r=0;r<reps;++r) acfun_conv1x1_int8(A.data(),M,K,Cout,B.data(),alpha.data(),beta.data(),cnew.data(),act); double tc=(now()-t0)/reps;

    printf("  M=%5d K=%4d N=%4d act=%d nt=%d  max|Δ|=%9.3e %s  gemm=%8.4fms  1x1=%8.4fms  %5.2fx\n",
           M,K,Cout,act,nt,md,ok?"OK":"FAIL",tg*1e3,tc*1e3,tg/tc);
    return ok;
}

int main(){
    printf("后端: %s  (1x1 专用内核 vs 通用 GEMM epilogue)\n", acfun_backend_name());
    int fails=0;
    /* 正确性矩阵 (单线程) */
    struct C{int m,k,n,a;}; C cs[]={
        {784,64,64,0},{784,64,64,1},{3136,32,32,1},{196,64,32,0},
        {50176,64,16,1},{64,64,17,1},{127,63,33,0},{1000,128,48,1},
    };
    printf("\n=== 正确性 (nt=1) ===\n");
    for(auto&c:cs) if(!run(c.m,c.k,c.n,c.a,1)) fails++;
    printf("\n=== 多线程一致性 (nt=4) ===\n");
    if(!run(3136,64,64,1,4)) fails++;
    if(!run(50176,64,32,1,4)) fails++;
    printf("\n%s\n", fails==0?"ALL PASS":(char*)"SOME FAILED");
    return fails==0?0:1;
}
