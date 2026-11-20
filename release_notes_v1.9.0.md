# Acfun v1.9.0 发行说明 — 小 K INT8 直接卷积多线程并行（224²×3 大 M 场景 2.53×，相对 float 反超到 1.8×）

> 本版在 v1.8.0（1×1 INT8 逐点卷积专用内核）之上，落实 README「下一步」明确列出的第一项：
> **小 K 直接卷积的多线程并行**。此前 `acfun_conv2d_int8_direct`（v1.7.0 引入）逐输出像素
> **完全串行**遍历 `n/oh/ow`，在全基准最大的第一层卷积场景 `224²×3→16 k3 s1`（K=27、M=50176）
> 上仅得 **1.0×**（INT8 2.087 ms vs float 2.034 ms），而对照的 float 卷积走 Winograd
> 内部已按 4 核配额并行——这是全基准最后一个「大 M、小 K」成片平推点。
>
> 本版把直接内核改为**按输出行 (n,oh) 展平的 OpenMP 静态并行**：每个像素输出区互不相交、
> 权重预打包 `Bp` / tap 解码表 / `alpha` / `beta` 全程只读共享、累加器栈上私有，零写冲突、
> 结果与线程数严格无关。工作感知在行数不足或输出过小时回落串行，避免 fork/join 拖累小图。
> `224²×3→16` INT8 从单线程 **2.826 ms 降到多线程 1.118 ms（2.53×）**，相对 float 由
> **1.0× 反超到 1.8×**；隔离测试跨 8 组边界 + 多线程逐元素 **max|Δ|=0**，全基准正确性全过，
> 其余路径零改动、零回归。

> 基底说明：v1.9.0 基于 v1.8.0（`c446d2c`，与 `origin/main` 同步，无分叉）。本版仅并行化
> 小 K INT8 直接卷积 AVX-512 内核并在基准中并列 1T/MT 口径；Winograd / 1×1 专用内核 /
> 通用 GEMM / epilogue / 量化遍 / BF16 / 各 float 算子一律零改动。

## 1. 为什么做（v1.9.0 的动机）

INT8 卷积经 v1.5–v1.8 消除了反量化遍、量化遍、1×1 与极小 K 的 im2col/打包固定开销后，
`224²×3→16 k3 s1`（Cin=3 < 4，不满足 Winograd 门槛）这类**极小 Cin 的 3×3** 必须走
小 K 直接卷积内核。该内核此前逐像素串行：

| 场景 | M=(OH·OW) | K | 线程 | INT8 耗时 | vs float |
|------|-----------|---|------|-----------|----------|
| `224²×3→16 k3 s1` | 50176 | 27 | 1T（旧） | 2.826 ms | ~1.0×（旧基准 2.087 ms） |
| 同上 | 50176 | 27 | 4T（本版） | **1.118 ms** | **1.8×** |

对照的 float 卷积走 Winograd，其内部 `acfun_effective_cores()` 已按 cgroup 配额（本机 4 核）
并行。直接内核的串行化是这一场景长期停留在 1.0× 的唯一结构性原因。v1.7.0 发行说明曾把
「小 K 直接内核多线程化」列为后续方向，本版予以兑现。

## 2. 本版做了什么

### 2.1 按输出行展平的 OpenMP 静态并行（`src/conv_int8_direct_avx512.cpp`）

沿用本仓库「SIMD 后端独立编译单元 + 运行时派发 + 标量兜底」范式，仅改造 AVX-512 VNNI 内核：

- **任务划分**：把 `n/oh/ow` 三层循环展平为 `rows = N·OH` 个输出行任务，每行内部仍按原顺序
  处理全部 `ow` 与输出通道块。`#pragma omp parallel num_threads(nt)` + `#pragma omp for
  schedule(static)`，静态连续分带、无动态队列锁竞争（与 float GEMM / Winograd 一致）。
- **零写冲突的正确性**：不同 `(n,oh)` 行写入 `output` 的互不相交区间；权重预打包 `Bp`、
  tap 解码表 `tap_ic/doh/dow`、`alpha/beta`、`inQ` 全程只读共享；每条 `(r, cb)` 的 int32
  累加器 `acc` 为栈上 `__m512i`，线程私有。因此结果与线程数**严格无关**。
- **工作感知回落**：`nt = acfun_get_num_threads_runtime()`（由调度器按 cgroup 配额下发，
  与 v1.8.0 1×1 内核 / GEMM 同构）；当 `rows < nt` 或输出元素 `< 65536` 时回落串行，
  避免 fork/join 与面板开销压垮小图（如 `56²×16 k1 s2`，实测 MT≈1.01×，零回退）。
- **数值不变**：内层 `vpdpbusd` 累加与 epilogue（显式 `mul`+`add`、无分支 `vmax` ReLU）
  逐字未改，`-ffp-contract=off` 保持与标量参考同舍入顺序。

### 2.2 隔离正确性回归（`tools/test_int8_direct.cpp`）

在原有「直接内核 vs im2col+标量 GEMM」逐元素比对基础上，新增**多线程一致性门**：
同一输入分别以 `nt=1` 与 `nt=4` 运行直接内核，要求 `max|Δ|(1T,4T)=0`，且两者均与参考
`max|Δ|<1e-5`。覆盖 8 组配置（1×1、3×3 padding、stride>1、非整除 Cout、batch>1、奇偶边界），
全部 `max|Δ|=0`。

### 2.3 基准口径升级（`src/benchmark.cpp`）

`bench_int8()` 的「INT8 量化卷积 vs float」段由单列 INT8 改为并列 **INT8-1T / INT8-MT**
两列，并给出 `MT加速` 与相对 float 的 `加速`。float 走 Winograd（内部 4 线程），INT8-MT
与之同口径公平对比；小图工作感知回落串行时 MT 加速≈1.0×。

## 3. 实测（沙箱 AMD EPYC 9K65 / AVX-512 VNNI，4 核 cgroup 配额；正确性门禁全过）

### 3.1 INT8 量化卷积：单线程 vs 多线程 vs float

| 配置 | FP(ms) | INT8-1T(ms) | INT8-MT(ms) | MT 加速 | vs float(MT) |
|------|--------|-------------|-------------|---------|--------------|
| `112²×16→16 k3 s1`（Winograd） | 1.039 | 1.365 | 1.358 | 1.01× | 0.8× |
| `56²×32→32 k3 s1`（Winograd） | 0.753 | 0.828 | 0.831 | 1.00× | 0.9× |
| `28²×64→64 k1 s1`（1×1 专用） | 0.114 | 0.092 | 0.060 | 1.54× | **1.9×** |
| `56²×16→32 k1 s2`（小 K，小图回落） | 0.055 | 0.084 | 0.083 | 1.01× | 0.7× |
| `224²×3→16 k3 s1`（小 K，大 M） | 2.018 | 2.826 | **1.118** | **2.53×** | **1.8×** |

> `224²×3→16` 为本版主攻场景：多线程把 INT8 从 2.826 ms 压到 1.118 ms，相对 float 由
> 约 1.0× 反超到 **1.8×**。墙钟绝对值在共享 cgroup 主机上有 ±20% 波动，方向与 2×+ 幅度稳定。
> Winograd / 1×1 路径本版未改；`28²×64` 的 MT 加速来自其 1×1 内核既有的按行并行在本基准
> 口径下的体现（旧基准将该项固定为 1T）。

### 3.2 正确性（最重要且可复现）

- 直接内核 1T / 4T / 标量参考 三者逐元素 **max|Δ|=0**（隔离测试 8 组全过）；
- 全基准「正确性：全部通过」，`FAIL` 计数 0。

### 3.3 其余路径（零回归）

GEMM 吞吐 / 卷积 / Winograd F(2,3)/F(4,3) / 深度卷积 / 分组卷积 / 最大池化 / FC+ReLU /
Post-GEMM Epilogue / INT8 GEMM / INT8 GEMM Epilogue / INT8 激活量化 / INT8 Winograd /
1×1 INT8 专用内核 / BF16 正确性与相对加速与 v1.8.0 持平；本版未触碰其任何代码。

## 4. 用法

API 不变。小 K INT8 卷积（`K=KH·KW·Cin≤64` 且非 1×1/s1/p0）自动进入直接内核，
其并行度由调度器下发的线程数决定：

```c
acfun_set_threads(4);                        /* 或 ACFUN_THREADS=4 */
acfun_conv2d_int8(input, N, H, W, Cin, wq, wscale, bias,
                  3, 3, Cout, 1, 1, s_in, output, act);
/* K=27 (Cin=3, 3x3) 自动走多线程小 K 直接内核; 小图自动回落串行。 */
```
```bash
./build/acfun_bench build/model.acfun        # 全基准 (INT8 卷积段并列 1T/MT)
# 隔离正确性回归 (含 1T/4T 一致性):
g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
    tools/test_int8_direct.cpp build/libacfun.a -fopenmp -o /tmp/test_int8_direct
/tmp/test_int8_direct                        # 期望 ALL PASS
```

## 5. 工程与文档

- `src/conv_int8_direct_avx512.cpp`：按输出行展平 OpenMP 并行 + 工作感知回落（`#include <omp.h>`）；
  数值内核逐字未改。
- `tools/test_int8_direct.cpp`：新增 1T/4T 逐元素一致性门。
- `src/benchmark.cpp`：INT8 卷积段并列 1T/MT 口径 + MT 加速列。
- `benchmark_result.txt`：全基准实测刷新。
- `README.md`：同步特性、路线与目录说明。

## 6. 下一步

- 小 K 直接内核的 tap 解码消除（复用 v1.8.0 1×1 的连续 A 加载思想扩展至 3×3 小 Cin）；
- 移动端 **ARM NEON** 后端（GEMM / 卷积 / 量化 / 池化）；
- 更细的**图优化**（算子重排 / 常量折叠 / Conv→Pool 跨层流式融合以彻底消去中间张量）。
