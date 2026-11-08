# Acfun v0.5.0 发行说明

> 本版汇集 `main` 上自 v0.4.0 以来的两段未发布开发成果，并统一作为 v0.5.0 发行：
> **AVX-512 Winograd 大幅提速** 与 **INT8 VNNI 量化推理引擎（实验性）**。
> 全部基准（`acfun_bench`）正确性通过。

## 1. AVX-512 Winograd 卷积大幅提速（本版速度重点）

延续「推理速度」目标，把 Winograd 卷积的 AVX-512 后端从「仅 256-bit 回落」升级为
**512-bit 向量化 + 寄存器驻留累加 + 瓦片并行**：

- **512-bit FMA 向量化**：最热累加内层 `m_hat[ij][c] += dh[ij]·g_hat[ij][d][c]` 一次覆盖 16 通道
  （较 AVX2 的 8 通道翻倍）；运行时派发顺序 **AVX-512 → AVX2 → 标量**。
- **寄存器驻留累加**：以「c 维 16 路分块」为最外层，每个 c 块用 16 个 zmm 累加器跨**全部**输入
  通道 `d` 累加、仅末了写回 `m_hat` 一次，消除旧路径「每 (d,ij) 重读/重写整条 m_hat」的冗余访存。
- **瓦片并行**：输出 2×2 瓦片对 `(ty,tx)` 网格做 OpenMP 并行，每线程私有 `m_hat`/`dh_all`，
  `g_hat`/`pad` 只读共享；小卷积（< 8M MAC）回落串行。
- **cgroup 感知线程上限** `acfun_effective_cores()`：读 `cpu.max` 配额取 `min(物理核, 配额)`，
  避免受限容器内多线程争抢导致的上下文切换风暴（曾使大卷积并行反而慢 2.7×）。
- **无分支 ReLU 融合**：输出写回归并 `vmax` 语义，消除数据相关分支惩罚——融合卷积由
  较分离路径更慢（0.8×）反转为更快（1.2×）。

### 实测（AVX-512，Zen5；与 v0.4.0 对比）

| 项目 | v0.4.0 | v0.5.0 | 提升 |
|------|--------|--------|------|
| Winograd 3×3 112²×16 | ~362 img/s（~3.9×） | **~498 img/s（~5.1×）** | **约 +38%** |
| 大卷积 112²×64 融合路径（4 核配额） | 9.4 ms | **7.8 ms（~3.5×）** | 约 -17% |
| 端到端 32³ 前向 | ~8800 img/s | ~8800 img/s | 持平 |

> 诚实边界：该卷积属**内存带宽受限**（`g_hat` 被每个瓦片重复读取），瓦片并行收益受 cgroup 配额 /
> 内存带宽约束；更大 Winograd 瓦片（F(4,3)/F(6,3)）以更少瓦片摊薄 `g_hat` 重取、进一步逼近 MNN
> 吞吐，是下一阶段方向。

## 2. INT8 VNNI 量化推理引擎（实验性，对标 MNN int8 路径）

补齐 x86 端 int8 吞吐的标准手法——`uint8×int8→int32` 的 `vpdpbusd` 累加。新增
`src/gemm_int8.cpp` / `src/gemm_int8_avx512.cpp`（独立 `-mavx512f -mfma -mavx512vnni` 编译单元）/
`src/conv_int8.cpp`：

- **INT8 GEMM 微内核**：`vpdpbusd` 单次对 4 行×4 列×连续 4 个 K 做点积并累加到 16 个 int32 累加器
  （64 个 int8 MAC/指令），算力密度较 float 提升 16×、带宽需求降为 1/4；扩展 4×4 面板布局打包为
  64 字节块，单条 `dpbusd` 即更新 4×4 块全部累加器。
- **关键调优**：B 列块驻留寄存器跨行块复用，把「每 dpbusd 负载比」从 2 降到 ~0.5，峰值由
  ~60–80 GOPS 抬升到 **~90 GOPS**；K 展开 + 列展开暴露指令级并行。
- **量化方案**：对称权重（逐通道 scale）+ 带零点（zp=128）逐张量激活，含零点偏置校正。
- **正确性（GEMM）**：与标量参考逐元素一致，**跨 60 组尺寸（含非整除边界）max|Δ|=0**。

> ⚠️ **INT8 卷积（实验性，已知差距）**：当前 `conv_int8` 走 im2col + INT8 GEMM，尚**无 INT8 Winograd**
> 快路径，且推理期逐元素 `round` 量化 / 反量化开销在「小卷积」上占主导，故 INT8 卷积**目前比 float
> 卷积慢**（例如 112²×16 约 79 img/s vs float ~499 img/s），随机权重下绝对误差也偏大（逐张量激活量化
> 最坏情形）。INT8 **GEMM 本身已正确且快**；把 INT8 卷积做到真正优于 float，需要 INT8 Winograd 快路径
> 与量化/反量化融合，列为下一阶段重点。

## 3. 其他

- 新增 `acfun_effective_cores()`（cgroup 感知线程上限）、无分支 ReLU 融合、INT8 量化/反量化 API。
- `CMakeLists.txt` 登记 `gemm_int8*.cpp` / `conv_int8.cpp` 为独立 SIMD 编译单元；`internal.h` 新增
  INT8 与 effective_cores 声明。
- `README.md` / `benchmark_result.txt` 已同步文档（AVX-512 Winograd + INT8 引擎）。
- 构建：`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j`；
  `python3 tools/gen_model.py build/model.acfun && ./build/acfun_bench build/model.acfun` 复现全部基准。
