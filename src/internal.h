#pragma once
/* Acfun 内部接口声明 (不对外暴露) */
#include <cstddef>
#ifdef __cplusplus
extern "C" {
#endif

/* 标量参考实现 (正确性基线) —— 方阵/整阵，行列主序步长均为 N */
void acfun_gemm_ref(int M, int N, int K,
                    const float* A, const float* B, float* C);

/* 带主维度(ldA,ldB,ldC)的通用标量实现，用于尾部余数子块 (beta=0, C 被覆盖) */
void acfun_gemm_ref_ld(int M, int N, int K,
                       int ldA, int ldB, int ldC,
                       const float* A, const float* B, float* C);
/* 同上但 beta=1: C += A*B (用于跨 KC 块累加的尾部余数) */
void acfun_gemm_ref_ld_b1(int M, int N, int K,
                          int ldA, int ldB, int ldC,
                          const float* A, const float* B, float* C);

/* SIMD 后端实现 (各自编译单元带对应 -march) */
void acfun_gemm_avx2(int M, int N, int K,
                     const float* A, const float* B, float* C);
void acfun_gemm_avx512(int M, int N, int K,
                       const float* A, const float* B, float* C);

/* 调度器下发的期望线程数 (1 = 串行); 后端据此选择串行或 omp parallel num_threads */
void acfun_set_num_threads_runtime(int n);
int  acfun_get_num_threads_runtime(void);

#ifdef __cplusplus
}
#endif
