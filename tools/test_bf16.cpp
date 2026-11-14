/* 独立 BF16 GEMM 验证: 正确性 (vs 标量参考) + 吞吐 (vs float GEMM) */
#include "acfun.h"
#include "internal.h"
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>

static double now(){ return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()*1e-9; }
static void rf(std::vector<float>& v, float s){ static std::mt19937 g(7); std::uniform_real_distribution<float> d(-s,s); for(auto&x:v)x=d(g); }

int main(){
    if(!__builtin_cpu_supports("avx512bf16")){ printf("NO avx512bf16\n"); return 1; }
    printf("=== BF16 GEMM 正确性 (vs 标量参考) ===\n");
    int fails=0;
    for(int S: {64,100,255,512,1023,2048}){
        int M=S,N=S,K=S;
        std::vector<float> A((size_t)M*K),B((size_t)K*N),Cr((size_t)M*N),Cb((size_t)M*N);
        rf(A,1.0f); rf(B,1.0f);
        acfun_gemm_ref(M,N,K,A.data(),B.data(),Cr.data());
        acfun_gemm_bf16(M,N,K,A.data(),B.data(),Cb.data());
        float md=0,norm=0;
        for(size_t i=0;i<Cr.size();++i){ md=std::max(md,std::fabs(Cr[i]-Cb[i])); norm=std::max(norm,std::fabs(Cr[i])); }
        float rel = norm>1e-9f? md/norm:0;
        bool ok = rel<0.02f; if(!ok) fails++;
        printf("  M=N=K=%4d  max|Δ|/范数=%7.3e %s\n", S, rel, ok?"OK":"FAIL");
    }
    printf("  正确性: %s\n", fails?"存在失败":"全部通过");

    printf("\n=== BF16 GEMM 吞吐 (单线程 & 4线程) vs float GEMM ===\n");
    printf("%-10s %12s %12s %11s %12s %12s %11s %11s\n","M=N=K","fp-1T","bf16-1T","bf16/FP","fp-4T","bf16-4T","b/f-4T","MTacc");
    for(int S: {256,512,1024,2048,4096}){
        int M=S,N=S,K=S;
        std::vector<float> A((size_t)M*K),B((size_t)K*N),Cf((size_t)M*N),Cb((size_t)M*N);
        rf(A,1.0f); rf(B,1.0f);
        double w=2.0*M*N*K, t;
        acfun_set_threads(1); acfun_set_num_threads_runtime(1);
        int reps=(S>=1024)?5:20;
        t=now(); for(int r=0;r<reps;++r) acfun_gemm(M,N,K,A.data(),B.data(),Cf.data()); t=(now()-t)/reps;
        double tf=w/t/1e9;
        acfun_set_num_threads_runtime(1);
        t=now(); for(int r=0;r<reps;++r) acfun_gemm_bf16(M,N,K,A.data(),B.data(),Cb.data()); t=(now()-t)/reps;
        double tb=w/t/1e9;
        acfun_set_threads(4);
        t=now(); for(int r=0;r<reps;++r) acfun_gemm(M,N,K,A.data(),B.data(),Cf.data()); t=(now()-t)/reps;
        double tf4=w/t/1e9;
        acfun_set_num_threads_runtime(4);
        t=now(); for(int r=0;r<reps;++r) acfun_gemm_bf16(M,N,K,A.data(),B.data(),Cb.data()); t=(now()-t)/reps;
        double tb4=w/t/1e9;
        printf("%-10d %12.1f %12.1f %10.1fx %12.1f %12.1f %10.1fx %10.1fx\n",
               S, tf, tb, tb/tf, tf4, tb4, tb4/tf4, tb4/tb);
    }
    return fails?1:0;
}
