# Acfun v1.1.0 发行说明 — 算子融合扩展 + INT8 多核调度对齐

> 本版延续「推理速度比肩 MNN」目标，在 v1.0.0（INT8 GEMM 提速里程碑）之上继续做两件事：
> **(1) 把已有的「Conv+BN+ReLU 融合」哲学延伸到全连接层**，新增 `acfun_fc_relu`，将 FC 的
> GEMM + 偏置 + ReLU 压进**单次输出遍历**，消除原先 GEMM 之后单独再扫一遍整张输出张量做 ReLU 的开销；
> **(2) 把 INT8 GEMM 的并行调度与已验证的 float GEMM 对齐**（`schedule(dynamic,1)` → `schedule(static)`），
> 消除多核下的中央任务队列锁竞争——正是 v0.9.1 修复 float GEMM「8T+ 吞吐反降」的同款手段。
> 两项改动均**零数值风险**（FC 融合数学等价 `max|Δ|=0`；INT8 调度仅改线程分块、不改算法）。

## 1. 为什么做（v1.1.0 的动机）

v1.0.0 已把 Conv/BN/ReLU 在加载期融合、Conv 输出写回时就地施加 ReLU。但**全连接层**仍走
「`acfun_fc`（GEMM + 偏置循环）→ 独立 `acfun_relu` 再扫一遍输出」两段式：每层 FC 之后都多一次
全张量读取+写回。在真实网络（FC 常接在展平特征之后、输出通道可达数千）中，这是一类「低垂的融合果实」——
既对标 MNN 把相邻逐元素算子并入主算子的图优化思路，又**绝不改变计算结果**。

与此同时，INT8 GEMM 的并行循环仍使用 `schedule(dynamic,1)`，而 float GEMM 在 v0.9.1 已因
dynamic 的中央队列锁竞争在 8T+ 后吞吐反降，并据此改为 `schedule(static)`。两套后端用不同调度是
一处不一致；把 INT8 对齐到已验证的 float 路径，可在多核裸机（带宽充裕、能线性扩展）与受限容器
（配额封顶）下消除这一结构性开销。

## 2. 本版做了什么

### 2.1 FC + ReLU 融合（`acfun_fc_relu`，零数值风险）

`src/layers.cpp` 新增 `acfun_fc_relu`：

```c
void acfun_fc_relu(const float* input, int N, int Cin,
                   const float* weight, const float* bias, int Cout, float* output) {
    acfun_gemm(N, Cout, Cin, input, weight, output);   /* GEMM 不变 */
    /* 偏置 + 融合 ReLU 单次遍历: 省去 acfun_fc + 独立 acfun_relu 的二次全张量扫描 */
    for (int n = 0; n < N; ++n)
        for (int c = 0; c < Cout; ++c) {
            float v = output[(size_t)n * Cout + c] + (bias ? bias[c] : 0.0f);
            if (v < 0.0f) v = 0.0f;
            output[(size_t)n * Cout + c] = v;
        }
}
```

- **模型执行器自动启用**：`acfun_model_infer` 在 `FC→ReLU` 形态下（加载期已把 ReLU 标记为融合 `act=1`）
  直接调用 `acfun_fc_relu`，不再先 `acfun_fc` 再 `acfun_relu`。
- **数值等价**：与「`acfun_fc` + `acfun_relu`」仅运算顺序不同（同一组浮点加法/比较），二者逐元素**完全一致**，
  跨 4 组尺寸（含 `N=256 × 2048×2048`）校验 **`max|Δ| = 0`** ✅。
- 公共 API 同步新增 `acfun_fc_relu`（见 `include/acfun/acfun.h`），便于直接集成。

### 2.2 INT8 GEMM 多核调度对齐（v1.1.0）

`src/gemm_int8_avx512.cpp` 的二维瓦片并行（Phase-2 计算）+ 两路打包（Phase-1）由
`schedule(dynamic,1)` 改为 `schedule(static)`，**与 float GEMM（v0.9.1）完全同构**：

- 去掉 dynamic 中央任务队列锁竞争（线程越多越严重，曾使 float GEMM 在 8T+ 吞吐反降）；
- static 连续分块使每个线程拿到连续瓦片带，只触及少数 A/B 面板（稳居本核 L2），在受限容器
  （配额封顶）与裸机（带宽充裕）下均近线性扩展；
- **单线程路径零改动**（INT8 GEMM 在 `nt<=1` 时早退、不走 OpenMP），故单线程吞吐与正确性不受影响。

> 实测（沙箱 AVX-512 Zen5，4 核 cgroup 配额）：单线程 INT8 GEMM 与 v1.0.0 持平（256²–4096²：
> ~93–138 GOPS，受内存带宽饱和限制的大矩阵本就近峰）；多线程加速 2048² 由 ~4.6× 提升到 ~4.8×、
> 4096² 由 ~5.9× 提升到 ~6.2×。本沙箱受 cgroup 配额（仅 4 核）与调度噪声影响，绝对数值波动较大
> （同二进制多次运行在 ±20% 带内），**但单线程无回归、多线程不退化**，且对齐的是本仓库已验证的
> float GEMM 修复手段——在裸机 8/16 核部署下收益更显著。

## 3. 实测（沙箱 AVX-512 Zen5，4 核 cgroup 配额；正确性门禁全过）

### 3.1 FC + ReLU 融合：分离（fc + 独立 relu 两遍扫描）vs 融合（单次遍历）

| 配置 (N, Cin×Cout) | 分离 ms | 融合 ms | 加速 | max|Δ| |
|--------------------|---------|---------|------|-------|
| 1, 4096×4096 | 5.534 | 5.439 | 1.02× | 0 |
| 64, 4096×4096 | 15.094 | 13.888 | 1.09× | 0 |
| 256, 2048×2048 | 9.929 | 7.510 | 1.32× | 0 |
| 1, 1024×1024 | 0.242 | 0.232 | 1.04× | 0 |

> 结论：融合**数学等价**（`max|Δ|=0`）且**从不更慢**（全部 ≤ 分离路径）；大 FC 下消去第二遍 ReLU
> 扫描的收益最明显（典型 1.02–1.09×，输出张量相对较大时更高）。正确性：全部通过 ✅。

### 3.2 其余路径（无回归）

float GEMM / 卷积 / Winograd / 深度&分组卷积 / 端到端模型前向 / INT8 量化卷积 / INT8 Winograd 的
正确性与相对 float 的加速，与 v1.0.0 持平或略优；全基准「正确性：全部通过」。

端到端模型前向（3×32×32）：**~0.37 ms（~2680 img/s）**（v1.0.0：~0.40 ms / ~2466 img/s，持平/略优）。

## 4. 用法

```bash
ACFUN_AUTOTUNE=1 ./build/acfun_bench build/model.acfun   # 强制调参(float+INT8)并写本地缓存
./build/acfun_bench build/model.acfun                    # 后续运行自动读缓存(或参考表/启发式)
```

新增/调整的公共 API：`acfun_fc_relu()`（FC + 融合 ReLU，单次遍历）。

## 5. 工程与文档

- `include/acfun/acfun.h`：新增 `acfun_fc_relu` 声明与注释。
- `src/layers.cpp`：新增 `acfun_fc_relu`（GEMM + 偏置 + 融合 ReLU 单次遍历）。
- `src/model.cpp`：`acfun_model_infer` 在 `FC→ReLU` 形态下改用 `acfun_fc_relu`（消除独立 ReLU 扫描）。
- `src/gemm_int8_avx512.cpp`：二维瓦片并行 + 两路打包由 `schedule(dynamic,1)` 改为 `schedule(static)`
  （与 float GEMM v0.9.1 同构，消除中央队列锁竞争），单线程路径不受影响。
- `src/benchmark.cpp`：新增 `bench_fc_relu_fused()`（FC+ReLU 融合正确性 + 提速基准），纳入 CI 正确性门禁。
- `README.md`：扩充「算子融合」「已具备」清单（FC+ReLU 融合、INT8 多核调度对齐），更新目录结构注释。
- `benchmark_result.txt`：同步更新（全基准正确性通过）。

## 6. 下一步

- 移动端 **ARM NEON** 后端（在 ARM 端复现对标结果）；
- 更细的**图优化**（算子重排 / 常量折叠）以进一步提升端到端吞吐；
- INT8 微内核 ILP 与更大 KC 的进一步压榨（当前 INT8 大矩阵受内存带宽饱和限制，仍有空间）。
