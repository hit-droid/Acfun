/**
 * Acfun — 高性能神经网络推理引擎
 * ---------------------------------------------------------------
 * 设计目标：以 MNN(阿里巴巴 Mobile Neural Network) 同款的工程手段
 * (分块 GEMM + SIMD 向量化 + 多线程 + 内存布局优化) 逼近其推理速度。
 *
 * 本头文件暴露 C 风格 API，便于跨语言与嵌入式集成。
 */
#ifndef ACFUN_ACFUN_H
#define ACFUN_ACFUN_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 线程数：传 0 表示使用全部逻辑核 */
void acfun_set_threads(int n);

/* 运行时查询当前后端：1=AVX512 2=AVX2 3=Scalar */
int acfun_backend(void);
const char* acfun_backend_name(void);

/* ---- 核心 GEMM: C[M,N] = A[M,K] * B[K,N] (row-major) ---- */
void acfun_gemm(int M, int N, int K,
                const float* A, const float* B, float* C);

/* ---- 卷积: NHWC 输入, im2col + GEMM ----
 * input : [N, H, W, Cin]
 * weight: [KH*KW*Cin, Cout]  (GEMM 权重, row-major)
 * bias  : [Cout] 或 NULL
 * output: [N, OH, OW, Cout]
 * stride / pad 为各维度统一值
 */
void acfun_conv2d(const float* input, int N, int H, int W, int Cin,
                  const float* weight, const float* bias,
                  int KH, int KW, int Cout, int stride, int pad,
                  float* output);

/* 卷积 + 融合 ReLU (输出写入时就地施加, 省去额外激活扫描) */
void acfun_conv2d_relu(const float* input, int N, int H, int W, int Cin,
                       const float* weight, const float* bias,
                       int KH, int KW, int Cout, int stride, int pad,
                       float* output);

/* 全连接: out[N, Cout] = in[N, Cin] * W[Cin, Cout] + bias */
void acfun_fc(const float* input, int N, int Cin,
              const float* weight, const float* bias, int Cout, float* output);

/* 逐元素激活 */
void acfun_relu(float* data, size_t len);
void acfun_relu6(float* data, size_t len);

/* 2D 最大池化 (NHWC, 方核, stride=pool) */
void acfun_maxpool(const float* input, int N, int H, int W, int C,
                   int pool, int stride, int pad, float* output);

/* Softmax (沿最后一维) */
void acfun_softmax(const float* input, int rows, int cols, float* output);

/* BatchNorm / 仿射: y = (x - mean)/sqrt(var+eps)*gamma + beta */
void acfun_batchnorm(float* data, int size, int channels,
                     const float* mean, const float* var,
                     const float* gamma, const float* beta, float eps);

/* ---- 模型执行 ----
 * 见 model.cpp 中定义的二进制格式 (magic "ACFUN0")。
 */
typedef struct AcfunModel AcfunModel;

AcfunModel* acfun_model_load(const char* path);
void acfun_model_destroy(AcfunModel* m);
/* 返回输出元素个数 */
int acfun_model_infer(AcfunModel* m, const float* input, float* output);
int acfun_model_output_size(AcfunModel* m);
/* 取模型输入维度 (batch 固定为 1) */
void acfun_model_input_dims(AcfunModel* m, int* Cin, int* Hin, int* Win);

#ifdef __cplusplus
}
#endif

#endif /* ACFUN_ACFUN_H */
