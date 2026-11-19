# Acfun v1.8.0 发行说明 — 1×1 INT8 逐点卷积专用内核（修复最后一片 0.3× 回退，全基准 INT8 再无回退点）

> 本版在 v1.7.0（小 K INT8 直接卷积内核，修复 224²×3 / 56²×16 k1s2）之上，攻下全基准
> **最后一片成片回退点**：1×1 / stride1 / pad0 的 INT8 卷积（`28²×64→64 k1 s1`）
> 长期慢于 float 卷积（仅 **0.3×**：INT8 0.403 ms vs float 0.117 ms）。这是自 v0.5.0
> 引入 INT8 以来，唯一在所有低精度优化版本（v1.5 epilogue / v1.6 量化向量化 /
> v1.7 小 K 直接内核）后仍持续回退的场景。
>
> 根因经逐阶段剖析确认：该场景 K=Cin=64、N=Cout=64、M=784，**量化遍仅 0.0018 ms，
> 通用 `acfun_gemm_int8_epi` 独占 0.396 ms（93%）**。三级分块 + 扩展 4×4 布局的
> packA/packB + 双缓冲 malloc + 微内核 GB×UC 展开初始化，这些固定开销远超
> 2·M·N·K ≈ 6.4 M MAC 的点积本身。v1.7.0 的小 K 直接内核虽能绕过打包，但其为通用
> 3×3 设计的 tap 解码（每 tap 3 次数组访问 + 边界判断 + 逐字节移位）在 1×1 上纯属冗余，
> 实测反导致 0.71 ms 回退，故当时被迫排除 1×1。
>
> 本版新增 **1×1 专用 VNNI 逐点内核**：A 行连续 4 字节一次加载并广播（无 im2col、无
> packA、无 tap 解码），权重预打包为 4k×16ch 布局跨全部 M 行复用，CB 个 int32 累加器
> 跨 k 常驻寄存器，融合 epilogue 用 16 通道掩码 load/store（消除通用 GEMM 的 4 次
> gather + 逐 lane 标量 scatter）。`28²×64 k1` 从 0.403 ms 降到 **0.092 ms（0.3× → 1.3×，
> 首次反超 float）**；隔离测试跨 8 组边界 + 多线程逐元素 **max|Δ|=0**，全基准
> 「正确性：全部通过」，其余路径零改动、零回归。

> 基底说明：v1.8.0 基于 v1.7.0（`e910527`，已与 `origin/main` 同步，无分叉）。本版仅新增
> 1×1 专用逐点内核并在 `conv_int8.cpp` 增加 1×1/s1/p0 路由；Winograd / 小 K 直接内核 /
> 通用 GEMM / epilogue / 量化遍 / 各 float 算子一律零改动、零回归。

## 1. 为什么做（v1.8.0 的动机）

INT8 卷积既有路径为「量化 → im2col → packA → INT8 GEMM + epilogue」。v1.5.0 / v1.6.0 /
v1.7.0 逐一消除了反量化遍、量化遍、小 K 的 im2col 开销后，**唯一残留的结构性瓶颈**是
1×1 卷积（无 im2col 可言）走通用 GEMM 的打包/分块固定开销。逐阶段实测（28²×64 k1，
单线程）：

| 阶段 | 耗时 | 占比 |
|------|------|------|
| 激活量化 `acfun_quantize_u8` | 0.0018 ms | <1% |
| INT8 GEMM `acfun_gemm_int8_epi` | 0.3960 ms | **93%** |
| 完整 `acfun_conv2d_int8` | 0.4249 ms | 100% |
| float 1×1 卷积参考 | 0.1170 ms | — |

对 K=64 的极小矩阵，通用 GEMM 的 `packA_int8`（把 A 重排为扩展 4×4 布局、放大 4×）、
`packB_int8`、`Apool/Bpool` 的 malloc、`GB×UC=16` 个累加器的初始化，以及 epilogue
写回时的 `_mm512_i32gather_ps`（4 组 alpha/beta）+ 逐 lane（16 次）条件 store，全部是
与点积规模无关的固定成本，彻底压倒了 6.4 M MAC 本身。

## 2. 本版做了什么

### 2.1 1×1 专用 VNNI 逐点内核

沿用本仓库「SIMD 后端独立编译单元 + 运行时派发 + 标量兜底」范式，新增三个编译单元/调度件：

- `src/conv1x1_int8_avx512.cpp`（`-mavx512f -mfma -mavx512vnni -ffp-contract=off`）：
  - **权重预打包**：`wq[K,Cout]` 一次重排为 VNNI「4-k × 16 通道」布局
    `Bp[cb,kg,lane,t]=wq[(kg·4+t)·Cout + cb·16+lane]`，跨全部 M 行复用（O(K·Cout)，
    相对 M·K·Cout 可忽略）；尾部 k/通道补 0。
  - **零拷贝连续 A**：1×1 的输入 `[M,K]` 本就连续，每行按 kg 一次 32-bit load 取连续
    4 个 k 输入字节（`std::memcpy` 到 uint32，规避 unaligned UB），`_mm_set1_epi32` +
    `_mm512_broadcast_i32x4` 广播到 16 lanes；**无 tap 解码、无边界判断、无逐字节移位**
    （这正是 v1.7.0 小 K 直接内核在 1×1 上回退的根因）。
  - **累加器驻留**：CB 个 `__m512i` 累加器跨全部 kg 常驻，一条 `vpdpbusd` 算完
    16 通道 × 4 个 k：`acc[ℓ] += Σ_t a4[t]·wq[(kg·4+t)·Cout + j0+ℓ]`。每行仅对 A
    做 M×Kg 次 load/broadcast，对全部输出通道块复用。
  - **融合 epilogue**：`acc(int32)→float ×alpha[j]+beta[j]`（显式 mul+add 非 FMA，与标量
    同舍入）+ 无分支 `vmax` ReLU，用 `_mm512_maskz_loadu_ps` / `_mm512_mask_storeu_ps`
    按 16 通道连续写回（消除通用 GEMM 的 gather + 逐 lane 标量 scatter）。
  - **多线程**：`acfun_get_num_threads_runtime()>1` 时按 M 行 `omp parallel for schedule(static)`
    并行，每线程独立累加器、无共享写冲突。
- `src/conv1x1_int8.cpp`：运行时按 `avx512vnni` 派发，否则标量兜底（可移植 + 正确性基线）。
  标量版逐行逐通道 k 升序累加，与 `acfun_gemm_int8_epi_ref` 同序同舍入。

### 2.2 路由与零回归设计（`conv_int8.cpp`）

- 专用内核只在 `KH==1 && KW==1 && stride==1 && pad==0` 时启用；该条件即 1×1 零拷贝
  GEMM 场景，A 连续、无 im2col、无 padding 语义，与内核假设严格一致。
- alpha/beta 计算原样复用（与 GEMM epilogue 共用同一份 scale 与零点校正），
  padding/量化语义不变；3×3/s1/p1 已被 Winograd 快路径截获、v1.7.0 小 K 内核处理其余
  小 K 非 1×1 场景，大 K 仍走原 GEMM —— 全部零改动。

### 2.3 隔离回归测试

- `tools/test_conv1x1.cpp`：把量化输入喂给通用 `acfun_gemm_int8_epi` 作参考，与专用内核
  逐元素比对。覆盖 M/K/Cout 多组（含非整除 64/17/33/48）、有/无 ReLU、nt=1/4 共 10 组，
  全部 `max|Δ|=0`。
- `src/benchmark.cpp` 新增 `bench_int8_1x1()` 段，纳入 CI 全基准长期回归。

## 3. 实测（沙箱 AMD EPYC 9K65 / AVX-512 VNNI，4 核 cgroup 配额；正确性门禁全过）

### 3.1 INT8 1×1 卷积：回退点转正（单线程，逐元素精度门 OK）

| 配置 (M×K×N) | 通用 GEMM | 1×1 专用内核 | 加速 | FP 卷积参考 |
|--------------|-----------|-------------|------|-------------|
| `784×64×64`（28²×64 k1 s1） | 0.477 ms | **0.080 ms** | **5.98×** | 0.117 ms |
| `784×64×64` +ReLU | 0.506 ms | **0.078 ms** | **6.49×** | 0.117 ms |
| `3136×32×32` | 0.648 ms | 0.090 ms | 7.19× | — |
| `50176×64×16` | 21.06 ms | 2.50 ms | 8.42× | — |
| `1000×128×48` | 1.180 ms | 0.159 ms | 7.41× | — |

全基准口径下 `28×28×64→64 k1 s1` INT8 从 **0.403 ms（0.3×）降到 0.092 ms（1.3×，
首次反超 float）**。墙钟绝对值在共享 cgroup 主机上有 ±20% 波动，方向稳定。

### 3.2 其余路径（零回归）

全基准「正确性：全部通过」，`FAIL` 计数 0。GEMM 吞吐 / 卷积 / Winograd F(2,3)/F(4,3) /
深度卷积 / 分组卷积 / 最大池化 / FC+ReLU / Post-GEMM Epilogue / INT8 GEMM / INT8 GEMM
Epilogue / INT8 激活量化 / INT8 Winograd / BF16 正确性与相对加速与 v1.7.0 持平。端到端
模型前向（3×32×32）~0.33 ms，与基线同量级。

## 4. 用法

API 不变，`acfun_conv2d_int8` 在 `KH==KW==1 && stride==1 && pad==0` 时自动走专用内核：

```c
acfun_conv2d_int8(input, N, H, W, Cin, wq, wscale, bias, 1, 1, Cout, 1, 0, s_in, output, act);
/* 自动派发 1x1 专用 VNNI 逐点内核, 其余路径原样。 */
```
```bash
./build/acfun_bench build/model.acfun        # 全基准（含新增 1x1 INT8 专用内核段）
# 隔离正确性回归:
g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
    tools/test_conv1x1.cpp build/libacfun.a -fopenmp -o /tmp/test_conv1x1
/tmp/test_conv1x1                            # 期望 ALL PASS
```

## 5. 工程与文档

- `src/conv1x1_int8.cpp` / `_avx512.cpp`：新增，1×1 专用逐点卷积调度器 + AVX-512 VNNI
  内核 + 标量兜底 + 多线程。
- `src/internal.h`：新增 `acfun_conv1x1_int8` 及两后端声明。
- `src/conv_int8.cpp`：新增 1×1/s1/p0 路由（在 im2col 构造前截获返回）。
- `CMakeLists.txt`：新增两个源文件及 AVX-512 VNNI 编译标志（含 `-ffp-contract=off`）。
- `tools/test_conv1x1.cpp`：新增隔离正确性回归测试。
- `src/benchmark.cpp`：新增 `bench_int8_1x1()` 基准段并注册到 main。
- `README.md` / `benchmark_result.txt`：同步更新特性、路线与基准（全基准正确性通过）。

## 6. 下一步

- 移动端 **ARM NEON** 后端（GEMM / 卷积 / 量化 / 池化）；
- 小 K 直接卷积 (`conv_int8_direct`) 的**多线程并行**与 tap 解码消除（复用本版 1×1 的
  连续 A 加载思想扩展至 3×3 小 Cin）；
- 更细的**图优化**（算子重排 / 常量折叠 / Conv→Pool 跨层流式融合以彻底消去中间张量）。
