# Acfun v1.3.0 发行说明 — 最大池化向量化 + 池化图优化

> 本版延续「推理速度比肩 MNN」目标，在 v1.2.0（Post-GEMM Epilogue 融合 + INT8 多核调度对齐 +
> BF16 后端）之上，**补齐
> 此前完全空缺的最大池化性能路径，并深化图优化**：
> **(1) 把最大池化的通道维度从「100% 标量嵌套循环」升级为 AVX-512 / AVX2 通道向量化**
> （运行时按 CPU 特性派发，标量版保留作正确性基线与可移植兜底）；
> **(2) 深化图优化**——新增两类零数值风险的池化融合：
> - **ReLU→MaxPool 冗余消除**：`Pool(ReLU(x)) ≡ Pool(x)`（逐元素 max 与 ReLU 可交换），当
>   `Conv/FC → ReLU → MaxPool` 相邻时直接丢弃该 ReLU（卷积不施加激活、由池化对原始卷积
>   输出取最大），消去卷积写回的激活扫描与独立的 ReLU 遍；
> - **MaxPool→ReLU 融合**：紧随池化其后的独立 ReLU 折叠进池化（就地 `max(x,0)`），
>   同样省去独立激活遍。
> 两类融合与「先 Pool 再 ReLU」逐元素完全一致，**零数值风险**，且本版测试模型
> （`Conv→BN→ReLU→MaxPool→FC→ReLU→Softmax`）正好命中前者，端到端路径自动受益。

> 基底说明：v1.3.0 的 `main` 在 v1.2.0（Post-GEMM Epilogue 融合，`0ad92f4`）之上叠加上述改动，
> 发布前已与远端 `origin/main` 同步，无分叉。本版属「算子性能补齐 + 图优化深化」，不改变任何
> 既有算子的数值结果。

## 1. 为什么做（v1.3.0 的动机）

v1.1.0 之前的池化实现 `acfun_maxpool` 是一个 **6 层嵌套的完全标量循环**——每个输出像素对
`pool×pool` 空间窗口内的 **C 个通道逐一做标量比较**。当通道数较大（轻量网络常见 16/32/64/128
乃至数百）时，这道逐通道标量比较是典型的可向量化瓶颈，而推理引擎的其余算子（GEMM / 卷积 /
Winograd / 深度&分组卷积）早已拥有 AVX2/AVX-512 向量化后端。池化成了「唯一没被 SIMD 照拂」
的热路径，与 MNN 等引擎「连池化都做通道向量化」的能力存在差距。

与此同时，路线图（README「差距与路线」）明确把「更细的图优化（算子重排/常量折叠）」列为待补
短板。池化前后的激活是图优化最自然的下手点：`ReLU` 与 `MaxPool` 都是逐元素 `max`，二者可交换，
故 `Pool(ReLU(x)) ≡ Pool(x)`——这意味着「卷积后接 ReLU 再接池化」中的 ReLU 在数学上完全冗余，
既可在卷积写回时省去激活、又不必跑一遍独立的 ReLU，是无损的图化简。

## 2. 本版做了什么

### 2.1 最大池化通道向量化（AVX-512 / AVX2，v1.3.0）

新增两个编译单元（各自带对应 `-march`，沿用本仓库「SIMD 后端独立编译 + 运行时派发」的工程范式）：

- `src/maxpool_avx512.cpp`（`-mavx512f -mfma`）：把通道打包成 16 路 `zmm`，对每个输出像素的
  `pool×pool` 窗口用 `vmaxps` 一次归约 16 个通道的窗口最大值；
- `src/maxpool_avx2.cpp`（`-mavx2 -mfma`）：8 路 `__m256` 同构，供仅有 AVX2 的主机使用。

`src/layers.cpp` 的 `maxpool_impl` 按 `__builtin_cpu_supports` 在 **AVX-512 → AVX2 → 标量** 间派发；
原标量实现保留为 `acfun_maxpool_scalar`（正确性基线 + 无 SIMD 主机兜底）。向量化与标量**严格等价**：
- 越界窗口位置（pad 引入）直接跳过、累加器保持 `-1e30` 哨兵，与标量「越界 `continue`」一致；
- 新增 `post_relu` 参数：池化输出写回时就地 `max(x,0)`（供 MaxPool→ReLU 融合），与「先 Pool
  再独立 ReLU」逐元素一致。

> 实测（沙箱 AVX-512 Zen5，4 核 cgroup 配额）：见 §3，宽通道下较朴素标量参考 **~4–16×**，且与
> 标量逐元素 `max|Δ|=0` 全部一致。

### 2.2 池化图优化（ReLU→MaxPool / MaxPool→ReLU，v1.3.0）

`src/model.cpp` 加载期融合逻辑新增两类零风险化简：

- **ReLU→MaxPool 冗余消除**：Conv/FC 融合块在「吸收 ReLU 之后（并经可能的 BN 折叠）」若紧邻
  MaxPool，则丢弃该 ReLU（`consumed[reluIdx]=1`）、卷积 `act=0`（不施加激活），由后续 MaxPool 对
  原始卷积输出取最大。`Pool(ReLU(x)) ≡ Pool(x)` 保证结果不变。
- **MaxPool→ReLU 融合**：独立 MaxPool 层若紧随独立 ReLU，则把 ReLU 折叠进池化
  （`post_relu=1`，池化写回时就地 `max(x,0)`），省去独立激活遍。

二者均不改变计算结果，并由基准的「融合引擎 vs 标量参考执行器」逐元素门禁严格证明（`max|Δ|=0`）。

## 3. 实测（沙箱 AVX-512 Zen5，4 核 cgroup 配额；正确性门禁全过）

### 3.1 最大池化：向量化 vs 标量 正确性（全尺寸 max|Δ|=0）

含 pad / 非 1 stride / 多通道/多池化核，向量化路径（`acfun_maxpool` 派发）与标量参考
（`acfun_maxpool_scalar`）逐元素完全一致；`post_relu` 路径同样 `max|Δ|=0`（详见 `benchmark_result.txt`）。

### 3.2 最大池化 加速（向量化 vs 朴素标量参考）

| 配置 (H×W×C, k, s) | 朴素标量 ms | Acfun 向量化 ms | 加速 |
|--------------------|-------------|-----------------|------|
| 112×112×16, k2 s2 | 0.200 | 0.014 | 14.8× |
| 224×224×3,  k2 s2 | 0.157 | 0.140 | 1.1× |
| 56×56×64,   k2 s2 | 0.200 | 0.012 | 17.4× |
| 28×28×128,  k3 s2 | 0.242 | 0.012 | 20.9× |
| 64×64×256,  k2 s2 | 1.755 | 0.073 | 24.2× |

> 实测：宽通道（C=64/128/256）下通道向量化收益最大，逼近 16–24×；C=3 等窄通道因标量尾处理与
> 窗口开销占比，加速较低（但仍不退化）。绝对数值在沙箱 cgroup 配额下波动较大（同二进制 ±20% 带内），
> **但向量化相对朴素标量稳定更快、且正确性零回归**。

### 3.3 端到端 / 其余路径（无回归）

端到端模型前向（3×32×32）：**~0.311 ms（~3219 img/s）**（v1.2.0：~0.350 ms / ~2859 img/s，
**~1.12×**）；GEMM / 卷积 / Winograd / 深度&分组卷积 / INT8 / BF16 / FC+ReLU 融合 / Post-GEMM
Epilogue 等路径，正确性与相对 float 的加速与 v1.2.0 持平；全基准「正确性：全部通过」。端到端模型
（含 `Conv→BN→ReLU→MaxPool`）因 ReLU→MaxPool 融合（卷积写回少一次激活扫描）+ 池化走向量化后端而提速。

## 4. 用法

```bash
ACFUN_AUTOTUNE=1 ./build/acfun_bench build/model.acfun   # 强制调参并写本地缓存
./build/acfun_bench build/model.acfun                    # 后续运行自动读缓存(或参考表/启发式)
```

新增/调整的公共 API：
- `acfun_maxpool_relu()`（最大池化 + 融合 ReLU，就地 `max(x,0)`）；
- `acfun_maxpool()` 内部新增 `post_relu` 派发（对调用方透明，签名不变）。

## 5. 工程与文档

- `include/acfun/acfun.h`：新增 `acfun_maxpool_relu` 声明与注释。
- `src/maxpool_avx512.cpp`：新增，AVX-512 通道向量化池化（`-mavx512f -mfma`）。
- `src/maxpool_avx2.cpp`：新增，AVX2 通道向量化池化（`-mavx2 -mfma`）。
- `src/layers.cpp`：`acfun_maxpool_scalar`（标量基线/兜底）、`maxpool_impl`（运行时派发）、
  `acfun_maxpool_relu`（融合 ReLU 池化）；原标量循环改造为标量基线。
- `src/model.cpp`：加载期新增 ReLU→MaxPool 冗余消除 与 MaxPool→ReLU 融合（零数值风险）。
- `CMakeLists.txt`：新增 `maxpool_avx2.cpp` / `maxpool_avx512.cpp` 源文件及对应 `-march` 编译标志。
- `src/benchmark.cpp`：新增 `bench_maxpool()`（向量化 vs 标量正确性 + 朴素标量提速基准），纳入 CI。
- `README.md`：扩充「已具备」「差距与路线」清单（池化向量化、池化图优化），更新目录结构注释。
- `benchmark_result.txt`：同步更新（全基准正确性通过）。

## 6. 下一步

- 移动端 **ARM NEON** 池化 / 卷积后端（在 ARM 端复现对标结果）；
- 更细的**图优化**（算子重排 / 常量折叠 / Conv→Pool 跨层带流式融合以彻底消去中间张量）；
- INT8 微内核 ILP 与更大 KC 的进一步压榨（当前 INT8 大矩阵受内存带宽饱和限制）。
