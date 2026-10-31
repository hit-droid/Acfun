# Acfun · 高性能神经网络推理引擎

> 目标：以 [MNN](https://github.com/alibaba/MNN)（阿里巴巴 Mobile Neural Network）同款的工程手段
> —— **分块 GEMM + SIMD 向量化 + OpenMP 多线程 + 内存布局优化 + 算子融合** —— 构建可对标其推理速度的引擎。

Acfun 当前聚焦**推理计算的核心：矩阵乘 (GEMM) 与卷积 (Conv)**，并以一套可序列化的模型格式驱动端到端前向。

---

## 设计要点

### 1. GEMM —— 推理速度的命脉
卷积、全连接在展开后都归结为 GEMM。Acfun 的 GEMM 采用业界成熟路线（GotoBLAS 风格三级分块）：

- **数据重排 (packing)**：A 面板按 `k` 连续重排，B 面板同样按 `k` 连续重排进连续缓冲区，使微内核沿 `k` 的 B 访问完全顺序化，**彻底消除对 B 的 `stride-N` 跨步访问**——这正是此前带宽利用率低的根因。
- **三级分块**：`MC`(命中 L3) × `KC`(命中 L2) × `NC`(命中 L1)。微内核工作集仅 `(MR+NR)·KC` 个寄存器 + 面板数据，稳居各级缓存，算术强度高。
- **微内核 (micro-kernel)**：AVX-512 后端为 **16×16**（`__m512`，满 512-bit 宽），AVX2 后端为 **8×8**；累加器跨整个 `KC` 累加，β=1 跨 KC 块累加保证大矩阵正确性。
- **多线程**：OpenMP 并行 **N 维**（每个 N 块的 B 面板被该线程内所有 M 块复用，提升缓存复用与负载均衡）；小矩阵自动退串行以消除线程池开销。
- **硬件自适应分块**：未显式指定时按 `sysconf` 探测的 **L2 容量**选取 `KC/NC`，使 A/B 面板合计稳居 L2（对标 MNN 的按硬件调参）；可用 `ACFUN_MC/KC/NC` 环境变量覆盖。
- **运行时派发**：二进制以通用指令集编译，运行时按 CPU 特性 (`avx512f` / `avx2`) 选择后端，保证可移植性。
- **数值兜底**：尾部余数（行/列不足分块）由标量参考实现补算，β=0/β=1 两套保证正确性。

### 2. 卷积 = im2col + GEMM
NHWC 输入经 `im2col` 展开为 `[OH·OW, KH·KW·Cin]` 矩阵，再复用 GEMM 内核，与 MNN 的 Conv 优化路线一致。

### 3. 算子融合（图优化）
加载期对计算图做融合，对标 MNN 的图优化：
- **BN 折叠**：紧跟 Conv/FC 的 BatchNorm 折叠进该层的权重与偏置，推理时**整层 BN 消失**（省一次全张量读写）。
- **ReLU 融合**：紧跟 Conv/FC 的 ReLU 在卷积输出写入时就地施加，**省去额外激活扫描**。
- 二者结合使 `Conv → BN → ReLU` 退化为单次融合卷积。

### 4. 推理层与模型
- 层：`Conv2D` / `FC` / `ReLU` / `MaxPool` / `Softmax` / `BatchNorm`。
- 模型格式 `ACFUN0`（小端二进制，见 `src/model.cpp`）：卷积/全连接含权重与偏置，FC 显式记录展平输入维，顺序执行器逐层前向。融合在加载期完成，模型文件保持原始层序。
- 工具 `tools/gen_model.py` 生成测试模型（随机权重，仅用于性能与流程验证）。

---

## 构建与测试

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 tools/gen_model.py build/model.acfun     # 生成测试模型
./build/acfun_bench build/model.acfun            # 正确性 + 性能基准
```

依赖：`cmake`、`g++`、`libomp-dev`（OpenMP）。

---

## 实测性能（本机：AMD EPYC 9K65，16 核 / 32 线程，每核 L2 1MB / L3 32MB 共享，AVX-512F+FMA）

> 共享多租户环境下多线程大矩阵数值会随邻居负载波动；单线程 GEMM 稳定逼近单核 FMA 峰值。

| 项目 | 数值 |
|------|------|
| 后端 | AVX-512 |
| GEMM 单线程 | **~96–118 GFLOPS**（≈ 该核 AVX-512 FMA 实用峰值 **90%+**） |
| GEMM 8 线程 | 典型 **200–500 GFLOPS**，随规模增长；较 naive 单线程 **10–20×** |
| GEMM vs 朴素实现 | 单线程 **~5×**，8 线程 **10–30×** |
| 卷积 3×3 (112²→112²) | 较 naive 直乘 **7–8×**（~580–680 img/s） |
| 端到端前向 (3×32×32) | **~8300 img/s** |
| Conv+BN+ReLU 融合 (112²×64) | 较三遍遍历基线 **1.4×**（max\|Δ\|< 6e-5） |
| 融合正确性 | 融合引擎 vs 非融合参考 **max\|Δ\|=4.5e-8**（数学等价） |
| 正确性 | 全尺寸（含非整除边界）`max\|Δ\| < 2e-5` ✅ |

> 调优甜点线程数由内存带宽决定（与 MNN 行为一致）：本机在 8 线程附近最佳；盲目堆砌线程会因带宽争用而收益递减。可用 `ACFUN_BENCH_THREADS` 覆盖演示线程数。

---

## 与 MNN 的对比（实事求是）

MNN 在移动端凭借 **ARM NEON + int8 量化 + 极致 autotuning** 达到极高吞吐。Acfun 目前验证了**同等工程方法论在 x86 上的可行性**，并已补齐多项关键能力：

- ✅ 已具备：分块 GEMM（B 打包 + 三级分块）、AVX2/AVX-512 双后端、运行时派发、im2col+GEMM 卷积、多线程、序列化模型、**Conv/BN/ReLU 算子融合**、**L2 感知的自动分块**、正确性校验。
- 🚧 差距与路线：移动端 **ARM NEON** 后端、**int8 量化**路径、按硬件的**极致自动调参 (autotuning)**、Winograd 等 3×3 专用算法、更细的图优化（算子重排/常量折叠）。

下一步以缩小上述差距、在 ARM 端复现对标结果为主。

---

## 目录结构

```
include/acfun/acfun.h   公共 C API
src/gemm.cpp            GEMM 调度器（运行时派发 + 线程控制）
src/gemm_ref.cpp        标量参考实现（β=0 / β=1，尾部余数）
src/gemm_avx2.cpp       AVX2+FMA 后端（8×8 微内核，B 打包 + 三级分块）
src/gemm_avx512.cpp     AVX-512F+FMA 后端（16×16 微内核，B 打包 + 三级分块）
src/layers.cpp          卷积（含融合 ReLU）、全连接、激活、池化、Softmax、BN
src/model.cpp           二进制模型加载（含 BN 折叠/ReLU 融合）与顺序执行器
src/benchmark.cpp       正确性 + GEMM/卷积/端到端/融合 基准
tools/gen_model.py      测试模型生成器
```

## License

[MIT](LICENSE)
