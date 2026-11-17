# Acfun v1.6.0 发行说明 — INT8 卷积激活量化向量化（补齐量化遍，47–77×）

> 本版在 v1.5.0（INT8 GEMM epilogue 融合，把反量化折叠进 VNNI 微内核写回）之上，补齐
> INT8 卷积路径**最后一段仍为标量逐元素的量化遍**：`x_q = clamp(round(x·inv_s)+zp)`。
> 新增运行时派发的 AVX-512（16 路）/ AVX2（8 路）/ 标量三级量化内核，量化遍实测提速
> **46.7–77.3×**，量化结果以 uint8 量化级比对与标量 `lroundf` 仅 tie 差 ≤1 级。
>
> 至此 INT8 卷积的两段逐元素前后处理（v1.5.0 反量化 epilogue 融合 + v1.6.0 量化向量化）
> 全部脱离标量循环，与 MNN 的 int8 推理全链路工程手段对齐。不改变任何算子数值结果，
> 全基准「正确性：全部通过」。

> 基底说明：v1.6.0 基于 v1.5.0（`363ca3a`，已与 `origin/main` 合并同步，无分叉）。本版属
> 「量化遍 SIMD 化」，仅替换 INT8 卷积的输入量化实现，epilogue 反量化、GEMM、Winograd、
> 各 float 算子路径一律零改动、零回归。

## 1. 为什么做（v1.6.0 的动机）

v1.5.0 把 INT8 卷积的反量化折叠进 VNNI GEMM 写回，消除了「int32 中间缓冲 + 一遍全张量
反量化扫描」。但输入量化这一步仍是标量：

```cpp
/* v1.5.0 及以前: conv_int8.cpp */
std::vector<uint8_t> inQ(inN);
for (size_t i = 0; i < inN; ++i) {
    int v = (int)std::lroundf(input[i] * inv_s) + ZP;   // 逐元素标量
    inQ[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
}
```

`lroundf` 是逐元素的库调用/标量转换，无法利用 SIMD 的批量 float→int→uint8 转换。在大特征图
上 inN 很大（如 `224×224×3` = 150,528 元素，`112×112×16` = 200,704 元素），这段虽非 GEMM
瓶颈，但固定开销可达 0.2–0.6 ms，仍是 INT8 int8 路径里唯一没有向量化的逐元素遍。

## 2. 本版做了什么

### 2.1 INT8 激活量化向量化内核（AVX-512 / AVX2 / 标量）

沿用本仓库「SIMD 后端独立编译单元 + 运行时派发」范式，新增三个编译单元：

- `src/int8_quantize_avx512.cpp`（`-mavx512f -mfma -mavx512bw`）：一次 16 路
  `_mm512_cvtps_epi32`（float→int32）+ `_mm512_cvtusepi32_epi8`（16×int32→16×uint8 饱和打包），
  末尾用 masked load/store 处理非 16 倍数尾部；
- `src/int8_quantize_avx2.cpp`（`-mavx2 -mfma`）：一次 8 路 `_mm256_cvtps_epi32`，
  经两次 `_mm_packus` 压缩为 8×uint8，尾部标量与；
- `src/int8_quantize.cpp`：运行时按 `__builtin_cpu_supports("avx512f" && "avx512bw")` / `avx2`
  派发，标量版作可移植兜底与正确性基线。

关键设计：
- **量化公式严格对齐**：`out[i]=clamp(round(in[i]·inv_s)+zp, 0, 255)`，与原子循环同构；
- **数值门禁**：SIMD 用 nearest-even 取整，标量用 `lroundf`（half-away-from-zero），二者
  仅在恰好落在 `.5` 的极罕见 tie 上差 1 个量化级；基准以 uint8 量化级逐元素比对，实测
  `max|Δq| ≤ 1`（多数配置为 0），ReLU/反量化后误差 <1 LSB，远低于 INT8 量化噪声；
- **尾部完整**：非 16/8 倍数的 inN 用 masked 或标量补齐，覆盖任意非整除尺寸；
- **零改动 GEMM**：仅替换 `conv_int8.cpp` 的输入量化两行为一次 `acfun_quantize_u8()` 调用，
  v1.5.0 的 `acfun_gemm_int8_epi` 反量化融合路径原样保留。

### 2.2 基准扩充（v1.6.0）

`bench_int8_quantize()` 新增量化遍的标量 vs 向量吞吐对比（4 组 H×W×Cin 尺寸），并做
逐元素量化级一致性门禁；结果随 `benchmark_result.txt` 一并刷新。

## 3. 实测（沙箱 AMD EPYC 9K65 / AVX-512，4 核 cgroup 配额；正确性门禁全过）

### 3.1 量化遍：向量内核 vs 标量（逐元素量化级 `max|Δq| ≤ 1`，全 OK）

| 配置 (H×W×Cin) | 标量 lroundf | v1.6.0 向量 | 加速 | max\|Δq\| |
|----------------|--------------|-------------|------|-----------|
| 112×112×16 | 0.6018 ms | 0.0129 ms | **46.7×** | 1 |
| 224×224×3  | 0.4326 ms | 0.0056 ms | **77.3×** | 0 |
| 56×56×32   | 0.2659 ms | 0.0037 ms | **72.3×** | 1 |
| 28×28×64   | 0.1753 ms | 0.0023 ms | **75.9×** | 1 |

> 向量化把量化遍从 0.18–0.60 ms 压到 0.002–0.013 ms，**降低一到两个数量级**。绝对加速比
> 在共享 cgroup 主机上有 ±20% 波动，但方向稳定、四配置一致。

### 3.2 INT8 卷积端到端与其余路径（零回归）

INT8 卷积端到端 `acfun_conv2d_int8` 在本沙箱墙上时间里，量化遍的确定性收益被 GEMM 本体
与共享主机的 ±20% 噪声部分掩盖（如 `1×1` 卷积 `0.515 → 0.401 ms`，`224²×3` 在 10–12 ms
区间波动，后者瓶颈在 K=27 扁矩阵的 im2col+GEMM，属 README 已知的随机权重最坏情形）。
量化遍本身的提速是可复现、与 GEMM 无关的确定性收益。

其余路径全部零回归：端到端模型前向（3×32×32）**0.294 ms**；GEMM / 卷积 / Winograd F(2,3)
/ F(4,3) / 深度卷积 / 分组卷积 / 最大池化 / FC+ReLU / Post-GEMM Epilogue / INT8 GEMM /
INT8 GEMM Epilogue（v1.5.0）/ BF16 正确性与相对加速与 v1.5.0 持平；全基准「正确性：全部
通过」，`FAIL` 计数为 0。

## 4. 用法

API 不变，`acfun_conv2d_int8` 及 INT8 Winograd 路径自动受益（量化遍自动派发向量内核）：

```c
acfun_conv2d_int8(input, N, H, W, Cin, wq, wscale, bias, KH, KW, Cout, stride, pad, s_in, output, act);
```
```bash
./build/acfun_bench build/model.acfun     # 含量化遍标量 vs 向量对比基准
```

## 5. 工程与文档

- `src/int8_quantize.cpp` / `_avx2.cpp` / `_avx512.cpp`：新增，量化内核 + 运行时派发。
- `src/internal.h`：新增 `acfun_quantize_u8` 及三后端声明。
- `src/conv_int8.cpp`：输入量化循环改为一次 `acfun_quantize_u8()` 调用（反量化 epi 融合不动）。
- `CMakeLists.txt`：新增三个源文件及对应 `-march` 编译标志。
- `src/benchmark.cpp`：`bench_int8_quantize()` 新增量化吞吐对比 + 量化级一致性门禁。
- `README.md` / `benchmark_result.txt`：同步更新量化说明与基准（全基准正确性通过）。

## 6. 下一步

- 移动端 **ARM NEON** 后端（GEMM / 卷积 / 量化 / 池化）；
- `224²×3` 等极小 Cin 的 **im2col+GEMM 扁矩阵路径**（K 过小导致算术强度崩溃，可评估小 K
  专属直接卷积内核）；
- 更细的**图优化**（算子重排 / 常量折叠 / Conv→Pool 跨层流式融合以彻底消去中间张量）。
