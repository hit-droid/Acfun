#pragma once
/* Acfun 内部接口声明 (不对外暴露) */
#include <cstddef>
#include <cstdint>
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

/* Winograd F(2,3) 卷积: 3x3 / stride1 / pad1, NHWC, 支持融合 ReLU (act=1) */
void acfun_winograd_conv3x3(const float* input, int N, int H, int W, int Cin,
                            const float* weight, const float* bias,
                            int Cout, float* output, int act);
/* 标量 (通用 -O3) 实现, 作为可移植兜底 */
void acfun_winograd_conv3x3_scalar(const float* input, int N, int H, int W, int Cin,
                                   const float* weight, const float* bias,
                                   int Cout, float* output, int act);
/* Winograd F(4,3) 卷积 (更大瓦片: 6x6 输入 / 4x4 输出 / 36 变换域元素), 对标 MNN 大瓦片路线:
 * 3x3 / stride1 / pad1, 仅依赖权重与输入块尺寸, 与 F(2,3) 同布局、同算法, 瓦片数降为 1/4。 */
void acfun_winograd43_conv3x3(const float* input, int N, int H, int W, int Cin,
                              const float* weight, const float* bias,
                              int Cout, float* output, int act);
void acfun_winograd43_conv3x3_scalar(const float* input, int N, int H, int W, int Cin,
                                     const float* weight, const float* bias,
                                     int Cout, float* output, int act);
void acfun_winograd43_conv3x3_avx512(const float* input, int N, int H, int W, int Cin,
                                     const float* weight, const float* bias,
                                     int Cout, float* output, int act);
/* AVX2+FMA 变体 (编译单元带 -mavx2 -mfma): 量化/累加内层 256-bit 向量化 */
void acfun_winograd_conv3x3_avx2(const float* input, int N, int H, int W, int Cin,
                                 const float* weight, const float* bias,
                                 int Cout, float* output, int act);
/* AVX-512+FMA 变体 (编译单元带 -mavx512f -mfma): 512-bit 量化/累加 + 寄存器驻留
 * 累加 (c 维 16 路分块, 跨全部 d 仅写回 m_hat 一次), 较 AVX2 翻倍通道吞吐。 */
void acfun_winograd_conv3x3_avx512(const float* input, int N, int H, int W, int Cin,
                             const float* weight, const float* bias,
                             int Cout, float* output, int act);

/* 逐通道(深度)卷积: 跨通道 16 路 AVX-512 向量化 (编译单元带 -mavx512f -mfma)。
 * weight 布局 [C, KH, KW]; 通用 -O3 标量版作可移植兜底。 */
void acfun_conv2d_dw_avx512(const float* input, int N, int H, int W, int C,
                            const float* weight, const float* bias,
                            int KH, int KW, int stride, int pad,
                            float* output, int act);
void acfun_conv2d_dw_scalar(const float* input, int N, int H, int W, int C,
                            const float* weight, const float* bias,
                            int KH, int KW, int stride, int pad,
                            float* output, int act);

/* 调度器下发的期望线程数 (1 = 串行); 后端据此选择串行或 omp parallel num_threads */
void acfun_set_num_threads_runtime(int n);
int  acfun_get_num_threads_runtime(void);

/* ---- INT8 量化 GEMM (对标 MNN int8 推理): A=uint8, B=int8, C=int32 累加 ----
 * 由 gemm_int8.cpp 调度器按 CPU 特性派发到 VNNI 后端或标量参考。 */
void acfun_gemm_int8(int M, int N, int K,
                     const uint8_t* A, const int8_t* B, int32_t* C);
/* 标量参考 (正确性基线), 含带主维度的变体用于尾部余数子块 (beta=0, C 被覆盖) */
void acfun_gemm_int8_ref(int M, int N, int K,
                         const uint8_t* A, const int8_t* B, int32_t* C);
void acfun_gemm_int8_ref_ld(int M, int N, int K,
                            int ldA, int ldB, int ldC,
                            const uint8_t* A, const int8_t* B, int32_t* C);
/* AVX-512 VNNI 后端 (编译单元带 -mavx512f -mfma -mavx512vnni) */
void acfun_gemm_int8_avx512(int M, int N, int K,
                            const uint8_t* A, const int8_t* B, int32_t* C);

/* INT8 Winograd F(2,3) 3x3 卷积 (整数域, 对标 MNN int8 路径): 3x3 / stride1 / pad1,
 * float 输入 (本函数内就地量化, 零点 zp=128), 逐通道 int8 权重。inv_s = 127/max|x|。
 * 由 conv_int8.cpp 在 3x3/s1/p1 且通道>=4 时自动路由, 省去独立的整张输入量化遍。 */
void acfun_winograd_conv3x3_int8(const float* input, int N, int H, int W, int Cin,
                                 const int8_t* wq, const float* wscale, const float* bias, int Cout,
                                 float s_in, float inv_s, float* output, int act);

/* ---- GEMM 自动调参 (autotuning): 运行时按硬件实测最优缓存分块 KC/NC ----
 * 返回 1 表示取到调参值(来自本地缓存或参考表), 0 表示无调参、回落静态启发式。
 * nt>1 取多线程调参, 否则取单线程调参。 */
int  acfun_get_tuned_kc_nc(int nt, int* KC, int* NC);
const char* acfun_tune_status(void);

/* 生效核心数: 取物理核数与 cgroup CPU 配额(cpu.max / cfs_quota)的较小值,
 * 避免多线程在受限容器内争抢导致上下文切换风暴 (例如 16 线程抢 4 核配额)。
 * 无 cgroup 限制时回退物理核数。供 GEMM/卷积并行区设置 num_threads 上限。 */
int acfun_effective_cores(void);

#ifdef __cplusplus
}
#endif
