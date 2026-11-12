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
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* 线程数：传 0 表示使用全部逻辑核 */
void acfun_set_threads(int n);

/* 运行时查询当前后端：1=AVX512 2=AVX2 3=Scalar */
int acfun_backend(void);
const char* acfun_backend_name(void);

/* GEMM 自动调参 (autotuning): 在目标机实测最优缓存分块 (KC/NC) 并写入本地缓存
 * (~/.cache/acfun/)。部署/换机时设置环境变量 ACFUN_AUTOTUNE=1 执行一次即可;
 * 调参后无需再设, 后续运行自动读取缓存(或命中内置参考表), 回落静态启发式。 */
void acfun_autotune(void);
const char* acfun_tune_status(void);

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

/* 逐通道(深度可分离)卷积: groups=Cin=Cout, 每个输出通道仅与其同名输入通道做 KH×KW 点积。
 * 权重布局 [C, KH, KW] (row-major, w[c*KH*KW + kh*KW + kw])。act=1 融合 ReLU。
 * 这是 MobileNet / EfficientNet 等轻量网络的核心算子, MNN 有专属深度卷积路径;
 * 相较把深度卷积当普通卷积走 im2col+GEMM (K=KH*KW 极小, 算术强度崩溃), 本路径
 * 跨通道 16 路 AVX-512 向量化, 带宽利用率大幅提升。 */
void acfun_conv2d_dw(const float* input, int N, int H, int W, int C,
                     const float* weight, const float* bias,
                     int KH, int KW, int stride, int pad,
                     float* output, int act);

/* 分组卷积: groups 组, 每组输入/输出通道 Cin/groups、Cout/groups。
 * 权重布局 [groups, Cout_per, KH*KW*Cin_per] (row-major, 组间连续)。
 * 当 groups==Cin==Cout 时自动路由到深度卷积 (acfun_conv2d_dw); 否则各组独立复用
 * 引擎既有 1x1 / Winograd / im2col 子路径, 与 MNN 的分组卷积实现一致。 */
void acfun_conv2d_g(const float* input, int N, int H, int W, int Cin,
                    const float* weight, const float* bias,
                    int KH, int KW, int Cout, int stride, int pad, int groups,
                    float* output, int act);

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

/* ---- INT8 量化卷积 (AVX-512 VNNI, 对标 MNN int8 推理) ----
 * 权重离线量化为 int8 (per-channel scale), 激活在线量化为 uint8 (per-tensor scale),
 * GEMM 内核以 vpdpbusd 将 uint8×int8 累加为 int32, 输出按 (s_in × 每通道 scale)
 * 反量化 (+bias, 可选 ReLU)。较 float 路径带宽节省 4×、算力密度提升 16×, 是 MNN
 * 在端侧/服务端拿吞吐的核心手段。权重需先经 acfun_quantize_weights 量化。 */
void acfun_quantize_weights(const float* w, int K, int N,
                            float* scales /*[N]*/, int8_t* wq /*[K*N]*/);
void acfun_conv2d_int8(const float* input, int N, int H, int W, int Cin,
                       const int8_t* wq, const float* wscale, const float* bias,
                       int KH, int KW, int Cout, int stride, int pad,
                       float s_in, float* output, int act);

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
