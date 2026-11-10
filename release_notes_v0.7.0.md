# Acfun v0.7.0 发行说明

> 本版延续「推理速度比肩 MNN」目标，补齐 **轻量网络（MobileNet / EfficientNet）落地的最后一块拼图——
> 深度可分离卷积与分组卷积**。新增 `acfun_conv2d_dw`（逐通道 AVX-512 跨通道 16 路向量化）与
> `acfun_conv2d_g`（通用分组卷积），并扩展模型格式 **type 7（ConvGrouped）** 与组感知 BN 折叠。
> 全部基准（`acfun_bench`）正确性通过，端到端模型前向与 Conv/BN/ReLU 融合对比恢复运行并全部 OK。

## 1. 深度卷积（Depthwise，`acfun_conv2d_dw`）

此前分组卷积只能当普通卷积走 `im2col+GEMM`（`K=KH·KW` 极小，算术强度崩溃），且 `groups==Cin==Cout`
的纯深度卷积根本无法正确表达——这是 Acfun 与 MNN 在轻量网络支持上的最大缺口。

- **权重布局**：`[C, KH, KW]`（row-major，`w[c*KH*KW + kh*KW + kw]`）。
- **AVX-512 跨通道 16 路向量化内核**（`src/dw_avx512.cpp`，`-mavx512f -mfma`）：
  - 权重预转置为 `wT[K·16]` 缓存于 L1；每个输出像素对 `KH·KW` 个空间偏移各做 16 路
    `_mm512_fmadd_ps`。
  - 尾通道与 padding 区域以 **AVX-512 掩码 load/store 归零**，消除全部边界分支与尾部特判。
- **派发**：`src/dw.cpp` 持标量参考实现，按 `__builtin_cpu_supports("avx512f")` 选 AVX-512，否则标量兜底。
- **实测**（沙箱 AVX-512，vs 标量参考，全部 `max|Δ| < 1e-6`）：
  112²×16 k3 **29.9×**、56²×32 k3 **33.8×**、64²×512 k3 **13.5×**、64²×64 k1 **35.4×**，
  极小图 16²×8 k3 仍 **13.3×**。大输入/大通道收益最高（13–36×）。

## 2. 分组卷积（Grouped，`acfun_conv2d_g`）

- **权重布局**：`[groups, Cout_per, KH·KW·Cin_per]`（row-major，组间连续）。
- **自动路由**：`groups==Cin==Cout`（纯深度）直接落入 §1 深度卷积路径（零额外开销）；否则按组把
  输入/输出抽取到连续临时缓冲后，对每组复用引擎既有 **1×1 / Winograd / im2col** 子路径——与 MNN 的
  分组卷积实现思路一致（NHWC 下各组通道交错，须先聚合成连续块）。
- **BN 折叠组感知**：`fold_bn_into` 按组索引（`gc=c/Cout_per`、`oc=c%Cout_per`、权重全局偏移
  `gc·Cout_per·per_in + t·Cout_per + oc`，并 **除以 `groups` 修正权重计数**），Conv/BN/ReLU 融合对
  分组层同样生效。
- **实测**（沙箱 AVX-512，vs 朴素分组参考，全部 `max|Δ| < 4e-6`）：
  56²×32→32 k3 g32 **32.3×**、56²×16→32 k3 g4 **2.5×**、28²×32→64 k1 g8 **3.7×**、14²×24→48 k3 g6 **1.7×**。

## 3. 模型格式扩展（type 7 = ConvGrouped）

- `src/model.cpp` 第一遍解析新增 type 7，携带 `groups` 字段；权重计数 `KH·KW·prevC·Cout / groups`
  （修复此前按普通卷积计数导致的 16× 越界读、损坏字节流的 bug）。
- `acfun_model_infer` 在 `groups>1` 时改调 `acfun_conv2d_g`；`tools/gen_model.py` 新增
  `conv_g(...)`（type 7，[Depthwise→BN→ReLU] 接入 Conv1 与 Conv2 之间，层数与字节布局同步修正）。
- 基准 `ref_parse`（参考执行器）同步修正同款权重计数 bug，端到端前向与融合对比恢复正常。

## 4. 工程与文档

- `CMakeLists.txt` 登记 `src/dw.cpp` / `src/dw_avx512.cpp`（后者带 `-mavx512f -mfma`）。
- `include/acfun/acfun.h` 新增 `acfun_conv2d_dw` / `acfun_conv2d_g` 公共 API 与完整注释。
- `src/benchmark.cpp` 新增 `bench_depthwise()` / `bench_grouped()`（对照朴素参考 + 加速比 + 正确性）；
  `src/layers.cpp` 的 `conv2d_impl` 增加 `groups` 形参与分组路由。
- `README.md` / `benchmark_result.txt` 同步新增深度卷积 / 分组卷积章节、性能表与目录结构条目；
  `tools/test_dw.cpp` 提供独立的深度/分组卷积正确性验证用例。

## 5. 已知边界（实事求是）

- 低频分组（g4/g8）受每组建模（临时缓冲抽取）开销影响，加速约 2–4×；纯深度卷积（g==Cin==Cout）加速最高。
- 深度卷积当前为逐通道独立 AVX-512 实现，未复用 1×1 快路径；超大通道（如 512）受每通道 9 次 FMA 的
  标量式循环主导，仍有进一步优化空间（输入打包 / 多通道并行点积）。
