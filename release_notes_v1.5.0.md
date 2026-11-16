# Acfun v1.5.0 发行说明 — INT8 GEMM Epilogue 融合（把反量化折叠进 VNNI 微内核写回）

> 本版在 v1.4.0（小通道组分组卷积 fused 直接内核）之上，落实了 `release_notes_v1.4.0.md`
> 明确列出的下一步之一：**epilogue 融合向 INT8 低精度 GEMM 后端延伸**。对标 MNN 在 INT8
> store 阶段做的融合，把 INT8 卷积此前「GEMM 写 int32 中间缓冲 → 独立一遍全张量反量化扫描」
> 的第二次内存往返彻底消掉，直接由 GEMM 微内核写出最终 float 结果。数学严格等价
> （`max|Δ|=0`），原 `acfun_gemm_int8` 路径零回归。

## 1. 为什么做（v1.5.0 的动机）

此前 INT8 卷积的 im2col+GEMM 路径（1×1、stride>1、Cin<4 的 3×3 等无法走 INT8 Winograd 的情形）
是「两段式」：

1. `acfun_gemm_int8` 把 uint8×int8 的点积写进一块 `int32[M·Cout]` 中间缓冲；
2. 再独立扫一遍整块输出：`v = i32·s_a·s_w[j] + bias[j] − s_a·s_w[j]·ZP·Sw[j]`，可选 ReLU。

第 2 遍是一次完整的「读 int32 → 反量化 → 写 float」全张量往返。对窄 N 卷积（GEMM 本体很廉价、
后处理占比高，如 `M=50176,N=16`）这一遍开销占比尤其可观。README 与 float 后端的经验
（`acfun_gemm_epi`）已经证明：MNN 在存储阶段做融合是拿吞吐的标准手段。INT8 路径此前是缺口。

## 2. 本版做了什么

### 2.1 新增 `acfun_gemm_int8_epi`（公共内部 API）

- `src/internal.h`：声明 `acfun_gemm_int8_epi` + 标量参考 `acfun_gemm_int8_epi_ref`
  + VNNI 后端 `acfun_gemm_int8_epi_avx512`。
- `src/gemm_int8.cpp`：标量参考（无 VNNI 主机的落地点）+ 运行时派发（按 `avx512vnni`）。
- `src/gemm_int8_avx512.cpp`：把原 `acfun_gemm_int8_avx512` 重构成内部 `gemm_int8_core(..., epi)`，
  原入口传 `epi=NULL` 走原 int32 写回（**零回归**），融合入口传真实描述符。分块 / 打包 /
  二维瓦片并行 / `schedule(static)` / cgroup 感知线程等全部复用，只有写回阶段分流。

签名：

```c
void acfun_gemm_int8_epi(int M, int N, int K,
                         const uint8_t* A, const int8_t* B, float* C,
                         const float* alpha, const float* beta, int act);
/* C[m,j] = (Σ_k A·B) · alpha[j] + beta[j]  (+ ReLU if act=1) */
```

- `alpha[j] = s_a · s_w[j]`：每输出通道反量化因子（激活 scale × 权重 scale）；
- `beta[j] = bias[j] − alpha[j] · ZP · Sw[j]`：把 bias 与 uint8 零点（ZP=128）偏置校正合并后的
  有效偏置（`Sw[j]=Σ_k w_q[k,j]`）。二者由调用方（卷积层）一次算好，长度 N。

### 2.2 微内核写回阶段的融合

`vpdpbusd` 累加器不变，融合只发生在 4×4 块写回：

1. `int32 → float`（`_mm512_cvtepi32_ps`）；
2. 按 lane 列索引 `L&3` 用 `_mm512_i32gather_ps` 取 `alpha[jcol+j] / beta[jcol+j]`；
   尾部不足 4 列时把越界 lane 索引钳到 `jR-1`（`jR≥1` 恒成立），避免 gather 读越界，
   这些 lane 在 scatter 时被有效列掩码过滤、不写回；
3. 显式 `mul` + `add`（`v = acc·alpha + beta`）反量化；
4. `act=1` 时用 `_mm512_max_ps(v, 0)` 无分支 ReLU；
5. scatter 写回 float 输出。

**严格等价的关键**：库中标量参考 `gemm_int8.cpp` 无 `-mfma`，是「乘后加」两次舍入；而 SIMD TU
带 `-mfma`，编译器默认 `-ffp-contract=fast` 会把分离的 mul+add 重新融合成 FMA（单次舍入），
造成 1 ulp 差。给该 TU 加 `-ffp-contract=off` 强制两次舍入后，SIMD 与标量 / 非融合参考
**逐元素 `max|Δ|=0`**。这一编译选项只影响这一处反量化（文件其余为整数 `vpdpbusd`，不受影响），
性能无实质变化。

### 2.3 接线 INT8 卷积 im2col 路径

`src/conv_int8.cpp` 的非 Winograd 分支删去 `int32[M·Cout]` 中间缓冲与独立反量化循环，改为预算
`alpha/beta` 后调用 `acfun_gemm_int8_epi` 一写到位；3×3 的 INT8 Winograd 快路径不受影响。

## 3. 实测（沙箱 AVX-512 Zen5，4 核 cgroup 配额）

### 3.1 正确性

- `acfun_gemm_int8_epi` vs「`acfun_gemm_int8` + 独立反量化（同 alpha/beta）」：`benchmark.cpp`
  内 7 组 + 独立工具 `tools/test_int8_epi.cpp` 共 25 组尺寸（含 `1023×4×7`、`7×1023×5` 等
  触发 gather 钳位与尾块的边界、有/无 ReLU、单/多线程）**全部 `max|Δ|=0`**；
- 标量参考 `acfun_gemm_int8_epi_ref` 与 SIMD 后端逐元素 `max|Δ|=0`；
- `acfun_bench` 全量基准 `FAIL=0`，`正确性: 全部通过` —— 原 GEMM / 卷积 / Winograd F2/F4 /
  1×1 / DW / 分组 / INT8 / BF16 / epilogue / 池化 / 端到端全部零回归。

### 3.2 吞吐（单线程，同进程 A/B 对比）

| M×N×K | 非融合(ms) | 融合 epi(ms) | 加速 |
|-------|-----------|--------------|------|
| 1024×1024×1024 | 30.746 | 23.932 | **1.28×** |
| 2048×2048×2048 | 214.196 | 189.171 | **1.13×** |
| 12544×16×144（卷积形） | 9.015 | 8.148 | **1.11×** |
| 50176×16×27（窄 N 卷积形） | 11.334 | 7.760 | **1.46×** |

> 诚实口径：大方阵里被消除的「一遍反量化扫描」相对 GEMM 本体较小，墙钟加速被 GEMM 计时噪声
> 部分掩盖（1.1–1.3×）；确定性、不随负载波动的收益是**删去 `M·N` 大小的 int32 中间缓冲**
> （省一次分配与峰值占用）与**一遍独立的全张量读改写**——在窄 N、内存受限部署下收益更清晰
> （50176×16×27 实测 1.46×）。与 float epilogue（v0.x）的结论同构。

## 4. 工程与文档

- `CMakeLists.txt`：`gemm_int8_avx512.cpp` 编译标志追加 `-ffp-contract=off`（严格等价）。
- `README.md`：更新「与 MNN 对比」清单与目录结构（新增 `tools/test_int8_epi.cpp`）。
- `benchmark_result.txt`：新增 INT8 epi 正确性 + 吞吐两段实测。
- 新增 `release_notes_v1.5.0.md`（本文件）与 `tools/test_int8_epi.cpp`（独立正确性工具）。

## 5. 下一步

- 把 epilogue 融合延伸到 **BF16 GEMM** 后端（`vdpbf16ps` 写回阶段叠加 bias + ReLU）；
- 探索 INT8 卷积 im2col 阶段与量化的进一步融合（消除一次量化输入扫描）；
- 移动端 **ARM NEON** 后端（在 ARM 端复现对标结果）；
- 更细的**图优化**（算子重排 / 常量折叠 / Conv→Pool 跨层流式融合）。
