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

/* 带融合后处理 (epilogue) 的标量参考: C[i,j] = Σ_k A·B + bias[j] (+ 可选 ReLU)。
 * 用作无 SIMD 主机 (backend==Scalar) 的 acfun_gemm_epi 落地点, 同时可作正确性基线。 */
void acfun_gemm_ref_epi(int M, int N, int K,
                        const float* A, const float* B, float* C,
                        const float* bias, int act);

/* SIMD 后端实现 (各自编译单元带对应 -march) */
void acfun_gemm_avx2(int M, int N, int K,
                     const float* A, const float* B, float* C);
void acfun_gemm_avx512(int M, int N, int K,
                       const float* A, const float* B, float* C);

/* 带融合后处理 (epilogue fusion: bias + 可选 ReLU) 的 GEMM 后端, 见 acfun_gemm_epi。
 * 仅当 bias 非 NULL 或 act!=0 时 (且为最后一个 KC 块) 才在写回时叠加后处理, 故
 * bias=NULL/act=0 时与无融合后端逐指令一致、零开销。 */
void acfun_gemm_avx2_epi(int M, int N, int K,
                         const float* A, const float* B, float* C,
                         const float* bias, int act);
void acfun_gemm_avx512_epi(int M, int N, int K,
                           const float* A, const float* B, float* C,
                           const float* bias, int act);

/* AVX-512 BF16 后端 (vdpbf16ps, 编译单元带 -mavx512f -mfma -mavx512bf16):
 * float 输入/输出, 内部转 bf16 计算、f32 累加; 带宽仅 float 之半, 对标 MNN bf16 低精度路径。 */
void acfun_gemm_bf16(int M, int N, int K,
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

/* ---- 分组卷积直接内核 (groups>1 且非深度) ----
 * 权重布局 [groups][KH*KW*Cin_per][Cout_per] (oc 连续), 与朴素参考/旧 im2col 路径一致。
 * 直接读 NHWC 输入、按输出通道向量化、直接写 NHWC 输出, 省去旧路径的逐组抽取/im2col/拷回。
 * 调度器按 Cout_per 与 CPU 特性在 AVX-512 / AVX2 / 标量间派发。 */
void acfun_conv2d_grouped(const float* input, int N, int H, int W, int Cin,
                          const float* weight, const float* bias,
                          int KH, int KW, int Cout, int stride, int pad,
                          int groups, float* output, int act);
void acfun_group_conv_scalar(const float* input, int N, int H, int W, int Cin,
                             const float* weight, const float* bias,
                             int KH, int KW, int Cout, int stride, int pad,
                             int groups, float* output, int act);
void acfun_group_conv_avx2(const float* input, int N, int H, int W, int Cin,
                           const float* weight, const float* bias,
                           int KH, int KW, int Cout, int stride, int pad,
                           int groups, float* output, int act);
void acfun_group_conv_avx512(const float* input, int N, int H, int W, int Cin,
                             const float* weight, const float* bias,
                             int KH, int KW, int Cout, int stride, int pad,
                             int groups, float* output, int act);

/* 调度器下发的期望线程数 (1 = 串行); 后端据此选择串行或 omp parallel num_threads */
void acfun_set_num_threads_runtime(int n);
int  acfun_get_num_threads_runtime(void);

/* 线程数选择 (工作感知 + cgroup 感知), 供 acfun_gemm / acfun_gemm_epi 共用 */
int  acfun_pick_threads(int M, int N, int K);

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

/* INT8 GEMM epilogue 融合 (对标 MNN store fusion): 把 INT8 卷积后处理
 *   C[m,j] = (Σ_k A[m,k]·B[k,j]) · alpha[j] + beta[j]  (+ReLU if act)
 * 折叠进微内核写回阶段, 直接写 float, 消除「GEMM 写 int32 中间缓冲 -> 独立一遍
 * 反量化/bias/零点校正/ReLU 全张量扫描」。alpha/beta 为长度 N 的每通道因子,
 * 由调用方把 激活scale×权重scale 折进 alpha, 把 bias 与零点校正折进 beta。
 * 数值上与「acfun_gemm_int8 写 int32 + 独立反量化扫描(同 alpha/beta)」严格等价。 */
void acfun_gemm_int8_epi(int M, int N, int K,
                         const uint8_t* A, const int8_t* B, float* C,
                         const float* alpha, const float* beta, int act);
/* 标量参考 (正确性基线, 亦为无 VNNI 主机的落地点) */
void acfun_gemm_int8_epi_ref(int M, int N, int K,
                             const uint8_t* A, const int8_t* B, float* C,
                             const float* alpha, const float* beta, int act);
/* AVX-512 VNNI 后端 (编译单元带 -mavx512f -mfma -mavx512vnni) */
void acfun_gemm_int8_epi_avx512(int M, int N, int K,
                                const uint8_t* A, const int8_t* B, float* C,
                                const float* alpha, const float* beta, int act);

/* INT8 卷积激活量化 (v1.6.0): O(inN) 逐元素遍, 此前为标量 lroundf 循环, 在大特征图上
 * 有明显固定开销。派发 AVX-512 (16路)/AVX2 (8路)/标量: out[i]=clamp(round(in[i]*inv_s)+zp,0,255)。
 * SIMD 用 nearest-even 取整, 与 lround 仅在恰好 .5 的 tie 上差 1 个量化级, 可忽略。 */
void acfun_quantize_u8(const float* in, uint8_t* out, size_t n, float inv_s, int zp);
void acfun_quantize_u8_scalar(const float* in, uint8_t* out, size_t n, float inv_s, int zp);
void acfun_quantize_u8_avx2(const float* in, uint8_t* out, size_t n, float inv_s, int zp);
void acfun_quantize_u8_avx512(const float* in, uint8_t* out, size_t n, float inv_s, int zp);

/* ---- 小 K INT8 直接卷积 (v1.7.0): 绕过 im2col + GEMM packA 的固定开销 ----
 * 针对「输入通道小 / 1x1 / stride>1」等 K=KH*KW*Cin 很小而 M 很大的回退场景
 * (如 224²×3→16、28²×64 1x1), 直接读 NHWC uint8 输入、按输出通道 16 路 VNNI
 * 累加、融合反量化/bias/ReLU 一遍写 float, 无中间缓冲。输入 inQ [N,H,W,Cin]
 * 已量化 (零点 zp=128, padding 亦填 zp), wq 为 [K,Cout] int8, alpha/beta 为
 * 每通道反量化因子/有效偏置 (由 conv_int8.cpp 预先算好, 与 GEMM 路径共用)。
 * 由 conv_int8.cpp 在小 K 且无 Winograd 快路径时派发。 */
void acfun_conv2d_int8_direct(const uint8_t* inQ, int N, int H, int W, int Cin,
                              const int8_t* wq, const float* alpha, const float* beta,
                              int KH, int KW, int Cout, int stride, int pad,
                              int OH, int OW, float* output, int act);
/* 标量兜底 (可移植): 逐像素逐通道标量累加, 数值为直接路径的正确性基线。 */
void acfun_conv2d_int8_direct_scalar(const uint8_t* inQ, int N, int H, int W, int Cin,
                                     const int8_t* wq, const float* alpha, const float* beta,
                                     int KH, int KW, int Cout, int stride, int pad,
                                     int OH, int OW, float* output, int act);
/* AVX-512 后端 (编译单元带 -mavx512f -mfma -mavx512vnni): 权重预打包为 4×16
 * VNNI 扩展布局 (一次, 跨全部输出像素复用), 逐输出像素 16 路通道 dpbusd 累加。 */
void acfun_conv2d_int8_direct_avx512(const uint8_t* inQ, int N, int H, int W, int Cin,
                                     const int8_t* wq, const float* alpha, const float* beta,
                                     int KH, int KW, int Cout, int stride, int pad,
                                     int OH, int OW, float* output, int act);

/* ---- 1x1 INT8 逐点卷积专用内核 (v1.8.0): 消除通用 GEMM 在小 K 上的打包/分块开销 ----
 * 1x1/stride1/pad0 的 A[M,K] 零拷贝连续 (K=Cin), 直接 VNNI 点积 + 融合 epilogue,
 * 无 im2col / packA / tap 解码。acc = Σ_k inQ·wq (k 升序), out = acc·alpha+beta (+ReLU),
 * 与 acfun_gemm_int8_epi 逐元素 max|Δ|=0。alpha/beta 由 conv_int8.cpp 预好共用。
 * 标量版为可移植兜底; AVX-512 VNNI 版权重预打包 4k×16ch 跨行复用、按行多线程。 */
void acfun_conv1x1_int8(const uint8_t* inQ, int M, int K, int Cout,
                        const int8_t* wq, const float* alpha, const float* beta,
                        float* output, int act);
void acfun_conv1x1_int8_scalar(const uint8_t* inQ, int M, int K, int Cout,
                               const int8_t* wq, const float* alpha, const float* beta,
                               float* output, int act);
void acfun_conv1x1_int8_avx512(const uint8_t* inQ, int M, int K, int Cout,
                               const int8_t* wq, const float* alpha, const float* beta,
                               float* output, int act);

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
/* INT8 GEMM 调参查询 (v1.0.0 新增): 返回 1 表示取到调参值(缓存或参考表), 0 表示回落启发式 */
int  acfun_get_tuned_kc_nc_int8(int nt, int* KC, int* NC);
const char* acfun_tune_status(void);

/* 最大池化: 通道向量化 (AVX-512 / AVX2) + 标量兜底, 运行时按 CPU 特性派发。
 * post_relu != 0 时在池化输出上就地施加 ReLU (与先 Pool 再独立 ReLU 数学等价, 二者
 * 仅运算顺序不同, 可合并到池化单遍, 省去独立激活扫描)。 */
void acfun_maxpool_scalar(const float* in, int N, int H, int W, int C,
                          int pool, int stride, int pad, float* out, int post_relu);
void acfun_maxpool_avx2(const float* in, int N, int H, int W, int C,
                        int pool, int stride, int pad, float* out, int post_relu);
void acfun_maxpool_avx512(const float* in, int N, int H, int W, int C,
                          int pool, int stride, int pad, float* out, int post_relu);

/* 生效核心数: 取物理核数与 cgroup CPU 配额(cpu.max / cfs_quota)的较小值,
 * 避免多线程在受限容器内争抢导致上下文切换风暴 (例如 16 线程抢 4 核配额)。
 * 无 cgroup 限制时回退物理核数。供 GEMM/卷积并行区设置 num_threads 上限。 */
int acfun_effective_cores(void);

#ifdef __cplusplus
}
#endif
