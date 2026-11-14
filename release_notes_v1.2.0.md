# Acfun v1.2.0 发行说明 — Post-GEMM Epilogue 融合（偏置 + ReLU 折叠进 GEMM 写回）

> 本版在 v1.1.0（算子融合扩展 + INT8 多核调度对齐，并已合并 BF16 后端）之上，补齐 MNN 在
> **存储阶段做融合**的标准手法：把 GEMM 输出阶段那一遍独立的「整体偏置加法 + ReLU」全张量读写
> 折叠进 GEMM 微内核的末尾写回。与 v1.1.0 的 `acfun_fc_relu`（层级的「FC + 偏置 + 就地 ReLU
> 单次遍历」）互补——后者在**层**边界消除一遍遍历，本版在**GEMM 内核**边界消除一遍遍历，二者
> 作用于不同层级、可叠加。

## 1. 为什么做（v1.2.0 的动机）

FC / Conv 在 GEMM 算完后，原本还要对整块输出 `C[M][N]` 再走一遍
`for (j) { c[j] += bias[j]; if (act) c[j] = max(c[j], 0); }`——一次完整的读改写。并且为了这遍融合，
im2col 卷积 / 1×1 卷积此前还额外分配了一块中间 `gemm_out` 缓冲（`A·B` 写入 `gemm_out`，再叠加
偏置 / ReLU 写回真正的 `output`）。

- **小 N（输出通道少）卷积的 GEMM 本体极廉价**：例如 3×3 / 112²、Cin=Cout=16 的 GEMM 形如
  `M=OH·OW、N=16、K=144`，仅约几十 MFLOP。这一遍独立全张量扫描的**访存开销占比反而可观**。
- **中间缓冲 `gemm_out` 是多余的显存与带宽**：它只是偏置/ReLU 融合的临时落脚点，完全可以省掉。

把偏置加法与 ReLU **折叠进 GEMM 微内核的末尾写回**，即可同时消除「二次全张量读写」与「中间缓冲」两笔开销。

## 2. 本版做了什么

### 2.1 新增公共 API `acfun_gemm_epi`（Post-GEMM Epilogue 融合）

```c
/* include/acfun/acfun.h */
void acfun_gemm_epi(int M, int N, int K,
                    const float* A, const float* B, float* C,
                    const float* bias, int act);
```

- `bias` 长度 N（逐输出列 / 逐通道），可为 `NULL`（不叠偏置）；`act` 取 0 / 1。
- 数值上与非融合路径**严格等价**：`bias` 仅在 `K` 维归约完成后叠加一次。

### 2.2 双后端落地融合（AVX-512 / AVX2）

- **仅在最后一个 KC 面板做 epilogue**：GEMM 对 `K` 方向是多面板累加，偏置必须在整行累加
  **完成后**才能加（否则每面板各加一遍 → 数值错误）。实现上令 `last = (k0+kc >= K)`，仅当 `last`
  时对该 16 列块施加 `c += bias` 与 `c = max(c, 0)`；`last` 之外的面板 epilogue 提前返回
  （`!bias && !act` 时零开销）。
- **寄存器内完成**：微内核在 FMA 累加后、store 前于寄存器里完成 `vadd(bias)` + `vmax(zero)`，
  省掉整块输出的二次访存。
- **`acfun_gemm_ref_epi`** 标量参考实现同步补齐，作为正确性基线与 Scalar 后端落点。

### 2.3 层侧接线（删中间缓冲 + 去独立偏置循环）

- `acfun_fc`：直接 `acfun_gemm_epi(..., bias, 0)`，**去掉原独立偏置循环**（注意：v1.1.0 的
  `acfun_fc_relu` 仍保留作为层级融合 API；`acfun_fc` 现走 GEMM 内核级 epilogue，二者互补）。
- `conv1x1_impl`：彻底删掉 `gemm_out` 中间缓冲，直接 `acfun_gemm_epi(..., output, bias, act)`。
- im2col 卷积路径：同样删掉 `gemm_out`，逐 batch 直接 `acfun_gemm_epi(..., output, bias, act)`。

既少一次全张量读写，又省一块 `M·N` 大小的内存分配与峰值占用。

### 2.4 派发与线程选择复用

- `acfun_gemm_epi` 与 `acfun_gemm` 共用同一套运行时后端派发（AVX-512 / AVX2 / 标量参考）与
  **工作感知线程选择** `acfun_pick_threads`（`acfun_pick_threads` 现由两条入口共用）。

## 3. 实测（沙箱 AVX-512 Zen5，融合 vs 未融合）

### 3.1 正确性（最重要且可复现）

`benchmark.cpp` 新增 `correctness_epi()`：以 `acfun_gemm_epi` 对标「`acfun_gemm` + 独立偏置 /
ReLU 参考」，跨 8 组尺寸（含 有/无偏置、有/无激活、非整除边界 64/100/255/512/63/1023/300/256）
`max|Δ| = 0`，数学严格等价。全部既有基准（Winograd F2/F4、1×1、深度&分组卷积、INT8、BF16、
端到端）正确性保持「全部通过」，零回归。

### 3.2 吞吐对比（共享 cgroup 主机，多次运行均值示意）

| 配置 | 未融合 | 融合 | 加速 |
|------|--------|------|------|
| GEMM 1024³（含 bias + ReLU） | ~9–13 ms | ~8–9 ms | **~1.0–1.6×**（噪声带内，多运行 ≈ 1.0–1.1×） |
| GEMM 64 / 100 / 255 / 512 等小尺寸 | — | 与未融合持平 | **~1.0–1.1×** |
| FC / 窄 N 卷积端到端 | 含独立扫描 + `gemm_out` | 一写到位 | 确定性内存/路径收益 |

> **诚实口径**：本沙箱为共享 cgroup，GEMM 计时在 ±20% 带内波动。被消除的偏置 + ReLU 整块二次访存
> （如 1024³ 输出 4MB，约 0.8 ms）相对 ~9 ms 的 GEMM 本体较小，故墙钟融合比在噪声带内测得
> **~1.0–1.1×**，个别运行（缺乏竞争时）曾达 ~1.6× 但不稳定。这是**测量噪声**，非正确性/架构问题。
>
> **确定性的真实收益**有两块，不受负载波动影响：① FC / 1×1 / im2col 卷积删去了 `gemm_out` 中间缓冲
> （少一次 `M·N` 大小的内存分配与峰值占用），利好显存受限部署；② 去掉一遍独立的整体偏置/ReLU 扫描，
> 代码路径更干净。在**裸机 / 带宽更宽裕**或**窄 N 卷积**（GEMM 更廉价、后处理占比更高）场景下，
> 这遍消除的墙钟收益会清晰体现。

## 4. 用法

```bash
./build/acfun_bench build/model.acfun    # 正确性(含 epilogue 段) + 性能基准
```

`acfun_gemm_epi` 为新增公共 API，引擎内部 FC / 卷积已默认走融合路径，调用方无需改动即可受益。

## 5. 工程与文档

- `include/acfun/acfun.h`：新增 `acfun_gemm_epi` 公共 API 与文档注释。
- `src/internal.h`：新增 `acfun_gemm_*_epi` 后端声明与 `acfun_pick_threads`。
- `src/gemm_ref.cpp`：新增 `acfun_gemm_ref_epi`（标量参考 + 正确性基线）。
- `src/gemm_avx512.cpp` / `src/gemm_avx2.cpp`：微内核末尾写回折叠偏置 + ReLU（仅最后 KC 面板生效）。
- `src/gemm.cpp`：抽取共享 `acfun_pick_threads`；新增 `acfun_gemm_epi` 派发。
- `src/layers.cpp`：`acfun_fc` / `conv1x1_impl` / im2col 卷积删去独立偏置循环与 `gemm_out` 缓冲，改走
  `acfun_gemm_epi`。
- `src/benchmark.cpp`：新增 `correctness_epi()`（8 组尺寸严格等价门禁）与 `bench_epi()`（6 组融合 vs
  未融合，修复尺寸标签打印）。
- `README.md`：新增「本轮新增：Post-GEMM Epilogue 融合」小节，并把该能力列入「已具备 ✅」清单。
- `benchmark_result.txt`：同步更新（含 epilogue 正确性段与融合基准段，autotune 缓存已启用）。
- 新增 `release_notes_v1.2.0.md`（本文件）。

## 6. 下一步

- 移动端 **ARM NEON** 后端（在 ARM 端复现对标结果）；
- 更细的**图优化**（算子重排 / 常量折叠）以进一步提升端到端吞吐；
- INT8 微内核 ILP 与更大 KC 的进一步压榨（当前 INT8 大矩阵受内存带宽饱和限制，仍有空间）；
- 探索 epilogue 融合向 INT8 / BF16 低精度 GEMM 后端延伸（同样可消整块输出的二次访存）。
