/* 标量 GEMM 参考实现：C[M,N] = A[M,K] * B[K,N] (row-major)
 * 用于正确性校验与无法向量化时的尾部余数计算。 */
#include "internal.h"

void acfun_gemm_ref_ld(int M, int N, int K,
                        int ldA, int ldB, int ldC,
                        const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + (size_t)i * ldA;
        float* c = C + (size_t)i * ldC;
        for (int j = 0; j < N; ++j) c[j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float av = a[k];
            if (av == 0.0f) continue;
            const float* b = B + (size_t)k * ldB;
            for (int j = 0; j < N; ++j) c[j] += av * b[j];
        }
    }
}

void acfun_gemm_ref(int M, int N, int K,
                    const float* A, const float* B, float* C) {
    acfun_gemm_ref_ld(M, N, K, K, N, N, A, B, C);
}

void acfun_gemm_ref_ld_b1(int M, int N, int K,
                          int ldA, int ldB, int ldC,
                          const float* A, const float* B, float* C) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + (size_t)i * ldA;
        float* c = C + (size_t)i * ldC;
        for (int k = 0; k < K; ++k) {
            float av = a[k];
            if (av == 0.0f) continue;
            const float* b = B + (size_t)k * ldB;
            for (int j = 0; j < N; ++j) c[j] += av * b[j];
        }
    }
}

/* 带融合后处理 (epilogue) 的标量参考: C[i,j] = Σ_k A[i,k]·B[k,j] + bias[j],
 * 若 act=1 再取 max(0,·)。后处理在 GEMM 归约完成后就地叠加, 与非融合路径严格等价。 */
void acfun_gemm_ref_epi(int M, int N, int K,
                        const float* A, const float* B, float* C,
                        const float* bias, int act) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + (size_t)i * K;
        float* c = C + (size_t)i * N;
        for (int j = 0; j < N; ++j) c[j] = 0.0f;
        for (int k = 0; k < K; ++k) {
            float av = a[k];
            if (av == 0.0f) continue;
            const float* b = B + (size_t)k * N;
            for (int j = 0; j < N; ++j) c[j] += av * b[j];
        }
        for (int j = 0; j < N; ++j) {
            float v = c[j];
            if (bias) v += bias[j];          /* 逐输出列 (通道) 偏置 */
            if (act && v < 0.0f) v = 0.0f;   /* 可选 ReLU */
            c[j] = v;
        }
    }
}
