# Acfun · 高性能神经网络推理引擎

> 目标：以 [MNN](https://github.com/alibaba/MNN)（阿里巴巴 Mobile Neural Network）同款的工程手段
> —— **分块 GEMM + SIMD 向量化 + OpenMP 多线程 + 内存布局优化** —— 构建可对标其推理速度的引擎。

Acfun 当前聚焦**推理计算的核心：矩阵乘 (GEMM) 与卷积 (Conv)**，并以一套可序列化的模型格式驱动端到端前向。

---

## 设计要点

### 1. GEMM —— 推理速度的命脉
卷积、全连接在展开后都归结为 GEMM。Acfun 的 GEMM 采用业界成熟路线：

- **数据重排 (packing)**：把 A/B 按 `k` 维度连续重排，使微内核每次读取均为顺序流，充分命中缓存。
- **微内核 (micro-kernel)**：AVX-512 后端为 **16×16**（`__m512`，满 512-bit 宽），AVX2 后端为 **8×8**；累加器常驻 SIMD 寄存器、跨整个 `K` 累加，算术强度高。
- **多线程**：OpenMP 并行 `M` 维度，细粒度分块；小矩阵自动退串行以消除线程池开销。
- **运行时派发**：二进制以通用指令集编译，运行时按 CPU 特性 (`avx512f` / `avx2`) 选择后端，保证可移植性。
- **数值兜底**：尾部余数（行/列不足分块）由标量参考实现补算，并通过整阵主维度 (`ldA/ldB/ldC`) 保证正确性。

### 2. 卷积 = im2col + GEMM
NHWC 输入经 `im2col` 展开为 `[OH·OW, KH·KW·Cin]` 矩阵，再复用 GEMM 内核，与 MNN 的 Conv 优化路线一致。

### 3. 推理层与模型
- 层：`Conv2D` / `FC` / `ReLU` / `MaxPool` / `Softmax` / `BatchNorm`。
- 模型格式 `ACFUN0`（小端二进制，见 `src/model.cpp`）：卷积/全连接含权重与偏置，FC 显式记录展平输入维，顺序执行器逐层前向。
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

## 实测性能（本机：32 核 / AVX-512F+FMA，8 线程演示）

| 项目 | 数值 |
|------|------|
| 后端 | AVX-512 |
| GEMM 单线程 | **~85 GFLOPS**（约为该内核实用峰值的 45%） |
| GEMM 8 线程 | **~440–640 GFLOPS**，较单线程 **4.9–7.5×** |
| GEMM vs 朴素实现 | 单线程 **~5–7×**，8 线程 **~30–40×** |
| 卷积 3×3 (112²→112²) | **10.4×** 朴素直乘（~435 img/s） |
| 端到端前向 (3×32×32) | **~5800 img/s** |
| 正确性 | 全尺寸（含非整除边界）`max|Δ| < 2e-5` ✅ |

> 调优甜点线程数由内存带宽决定（与 MNN 行为一致）：本机在 8 线程附近最佳；盲目堆砌线程会因带宽争用而收益递减。可用 `ACFUN_BENCH_THREADS` 覆盖演示线程数。

---

## 与 MNN 的对比（实事求是）

MNN 在移动端凭借 **ARM NEON + int8 量化 + 极致 autotuning** 达到极高吞吐。Acfun 目前验证了**同等工程方法论在 x86 上的可行性**：

- ✅ 已具备：分块 GEMM、AVX2/AVX-512 双后端、运行时派发、im2col+GEMM 卷积、多线程、序列化模型、正确性校验。
- 🚧 差距与路线：移动端 **ARM NEON** 后端、**int8 量化**路径、按硬件的**自动调参 (autotuning)**、算子融合（Conv+BN+ReLU）与图优化。

下一步以缩小上述差距、在 ARM 端复现对标结果为主。

---

## 目录结构

```
include/acfun/acfun.h   公共 C API
src/gemm_*.cpp          GEMM（标量参考 / AVX2 / AVX-512 / 派发）
src/layers.cpp          卷积、全连接、激活、池化、Softmax、BN
src/model.cpp           二进制模型加载与顺序执行器
src/benchmark.cpp       正确性 + 吞吐基准
tools/gen_model.py      测试模型生成器
```

## License

[MIT](LICENSE)
