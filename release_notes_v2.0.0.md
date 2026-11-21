# Acfun v2.0.0 发行说明 — 小 K INT8 直接卷积 tap 解码消除（224²×3 单线程 2.3×，多线程 3.0×）

> 本版在 v1.9.0（小 K 直接卷积多线程并行）之上，落实 README「下一步」长期列出的首项：
> **小 K 直接内核的 tap 解码消除（复用 1×1 专用内核的连续 A 加载思想扩展至 3×3 小 Cin）**。
> v1.7.0 引入的直接内核虽避免了 im2col/packA 双重放大，但仍在每个输出像素的最内层
> 重复执行标量 tap 解码：每个 tap 做 3 次整数乘法算地址 `(((size_t)ih*W+iw)*Cin+ic)`、
> 4 次越界比较，且 4 个 k 字节的 a4 打包在输出通道块（cb）循环**内部**被重复解码 CB 次
> （Cout=32/64 时重复 2–4 倍）。在 `224²×3→16 k3`（K=27、M=50176）这类大 M 场景，
> 这段标量开销是点积之外的头号瓶颈。
>
> 本版做三件**数值严格等价**（逐元素 `max|Δ|=0`）的改造：
> ① 预计算每个 tap 相对像素中心的**字节偏移** `byteoff[k]=(doh*W+dow)*Cin+ic`，
> 把最内层的 3 次整数乘法降为 1 次加法；
> ② 为每个输出行/列判定「内部像素」（感受野完全落在图内），走无边界判断的快速解码分支
> （224² 场景 98% 像素为内部像素）；
> ③ 把 a4 的 4 字节解码提到 cb 循环**外**，一次解码跨全部输出通道块复用。
>
> 内核级稳定 A/B（`tools/bench_int8_direct.cpp`，30 轮取中位数，两轮独立运行一致，
> checksum 逐字节相同）：`224²×3→16` 单线程 **2.724→1.183 ms（2.30×）**、多线程
> **1.026→0.338 ms（3.03×）**；`56²×16→32 k1 s2` 单线程 **0.053→0.015 ms（3.53×）**，
> 相对 float 由 0.7× 回退转为 **正收益**。隔离测试跨 8 组边界 1T/4T 与标量参考逐元素
> `max|Δ|=0`，全基准正确性全过，其余路径零改动、零回归。

> 基底说明：v2.0.0 基于 v1.9.0（`62b7809`，= `origin/main`，无分叉）。本版仅改造
> `src/conv_int8_direct_avx512.cpp` 一个编译单元的 tap 解码路径；Winograd / 1×1 专用内核 /
> 通用 GEMM / epilogue / 量化遍 / BF16 / 各 float 算子一律零改动。新增独立微基准工具
> `tools/bench_int8_direct.cpp`（不影响库行为）。

## 1. 为什么做（v2.0.0 的动机）

README「与 MNN 的对比」长期把「小 K 直接内核的 tap 解码消除」列为差距与路线首项。
v1.7.0 消除了 im2col/packA 固定开销、v1.9.0 消除了串行化，但直接内核最内层仍有两个
结构性冗余（均在 `src/conv_int8_direct_avx512.cpp`）：

| 冗余 | 旧代码 | 量级 |
|------|--------|------|
| tap 寻址 3 次乘法 | `inb[((size_t)ih*W+iw)*Cin+tap_ic[k]]`，`ih/oh*stride+doh`、`iw=ow*stride+dow` | 每 tap 3 乘 2 加 |
| 边界判断不可省 | 即使像素感受野完全在图内，仍逐 tap 做 `ih>=0&&ih<H&&iw>=0&&iw<W` | 每 tap 4 比较 |
| a4 解码在 cb 内 | 同一像素同一 kg 的 4 字节 a4 在每个输出通道块重算 | 重复 CB 次（Cout=32→2×，64→4×） |

对 `224²×3→16`（M=50176、Kg=7、CB=1），前两项 dominate；对 `56²×16→32 k1s2`（CB=2），
第三项亦生效。这是 float 卷积（Winograd 已向量化 + 并行）能把 INT8 甩在身后的最后一片
标量热点。

## 2. 本版做了什么

### 2.1 tap 字节偏移预计算（消除最内层乘法）

沿用「SIMD 后端独立编译单元 + 运行时派发 + 标量兜底」范式，仅改造 AVX-512 VNNI 内核：

- 在 tap 预解码表一次生成 `tap_byteoff[k] = (tap_doh[k]*W + tap_dow[k])*Cin + tap_ic[k]`
  （全图只读共享，K≤64 故 64 项栈数组）。
- 取数地址由 `((size_t)ih*W+iw)*Cin+tap_ic[k]` 变为 `base + tap_byteoff[k]`，其中
  `base=(oh*stride*W+ow*stride)*Cin` 每像素算一次。**代数严格等价**，省去每 tap 3 次乘法。

### 2.2 内部像素无边界判断快速路径

- 预扫描全部 tap 得 `min/max doh/dow`；对每个输出行 `ohs=oh*stride`、列 `ows=ow*stride`
  判定 `[ohs+min_doh, ohs+max_doh]⊂[0,H)` 且 `[ows+min_dow, ows+max_dow]⊂[0,W)`。
- 满足时该像素全部 tap 必在图内，走快速分支直接 `inb[base+tap_byteoff[k]]`，跳过 4 次边界比较；
  否则走原边界分支（padding 取 zp=128）。**两分支取同一地址、同一 xq**，结果一致。

### 2.3 a4 解码提到通道块循环外

- 把「当前像素全部 Kg 个 a4 的 4 字节打包」提到 `for cb` 之前，存到栈上 `a4buf[Kg]`；
  cb 循环内只做 `_mm512_broadcast_i32x4` + `vpdpbusd` + epilogue。**数值不变**，
  a4 解码次数从 `CB×Kg×4` 降为 `Kg×4`。

### 2.4 稳定 A/B 微基准（`tools/bench_int8_direct.cpp`）

新增独立微基准：对直接内核做 30 轮计时取中位数，只测内核（输入/权重预先量化），
分别链接 v1.9.0 基线库（git worktree @ `HEAD`）与 v2.0.0 优化库。checksum 逐字节一致
证明两版输入完全相同，消除共享主机墙钟抖动对加速比结论的干扰。

## 3. 实测（沙箱 AMD EPYC 9K65 / AVX-512 VNNI，4 核 cgroup 配额；正确性门禁全过）

### 3.1 内核级稳定 A/B（30 轮中位数，两轮独立运行一致）

| 场景 | 线程 | v1.9.0(ms) | v2.0.0(ms) | 加速 |
|------|------|-----------|-----------|------|
| `224²×3→16 k3 s1 p1` | 1T | 2.724 | **1.183** | **2.30×** |
| `224²×3→16 k3 s1 p1` | 4T | 1.026 | **0.338** | **3.03×** |
| `56²×16→32 k1 s2 p0` | 1T | 0.053 | **0.015** | **3.53×** |
| `20²×3→13 k3 s2 p1` | 1T | 0.006 | **0.003** | ~2.0× |

> 224²×3 为本版主攻场景：单线程 2.30×、多线程 3.03×。多线程加速比高于单线程，是因为
> tap 解码优化后每核工作量下降，并行有效负载占比更高。`56²×16 k1s2`（旧 0.7× 回退点）
> 内核从 0.053 降到 0.015 ms，端到端已由回退转正收益。

### 3.2 端到端 INT8 卷积段（本版优化的两行，更新至 benchmark_result.txt）

| 配置 | FP(ms) | INT8-1T(ms) | INT8-MT(ms) | vs float(MT) |
|------|--------|-------------|-------------|--------------|
| `56²×16→32 k1 s2`（**本版优化**） | 0.079 | **0.053** | **0.055** | **1.4×** |
| `224²×3→16 k3 s1`（**本版优化**） | 2.038 | **1.241** | **0.437** | **4.7×** |

> 对照 v1.9.0 基线（`benchmark_result.txt` 历史值）：`56²×16` INT8-1T/MT 由 0.084/0.083 ms
> 降到 0.053/0.055 ms，相对 float 由 **0.7× 回退转 1.4× 正收益**；`224²×3` INT8-1T/MT 由
> 2.826/1.118 ms 降到 1.241/0.437 ms。其余三行（Winograd / 1×1 专用路径）本版未改、保留
> 基线值。墙钟单次有 ±20% 抖动，加速比结论以 3.1 节的 30 轮中位数 A/B 为准。

### 3.3 正确性（最重要且可复现）

- 直接内核隔离测试 `tools/test_int8_direct.cpp` 跨 8 组配置（1×1 / 3×3 padding / stride>1 /
  非整除 Cout / batch>1 / 奇偶边界 / 有无 ReLU）：与 im2col+标量 GEMM 参考逐元素 `max|Δ|=0`，
  且 1T/4T 逐元素 `max|Δ|=0`。
- 全基准「正确性：全部通过」，`FAIL` 计数 0。
- 稳定 A/B 的 checksum 新旧逐字节相同，证明优化未改变任何输出。

### 3.4 其余路径（零回归）

GEMM 吞吐 / 卷积 / Winograd F(2,3)/F(4,3) / 深度卷积 / 分组卷积 / 最大池化 / FC+ReLU /
Post-GEMM Epilogue / INT8 GEMM / INT8 GEMM Epilogue / INT8 激活量化 / INT8 Winograd /
1×1 INT8 专用内核 / BF16 正确性与相对加速与 v1.9.0 持平；本版未触碰其任何代码。

> **关于 CI**：同 v1.9.0，GitHub ubuntu-latest runner（2 核，无 AVX-512 VNNI）上
> `correctness_int8_epi` 7 组的 `max|Δ|=1.221e-4` 为历史遗留（该环境无等效舍入路径），
> 与版本无关；本版改动的 INT8 直接卷积相关正确性项在该 runner 上全部 OK。

## 4. 用法

API 不变。小 K INT8 卷积自动进入直接内核并按本版优化路径执行：

```c
acfun_set_threads(4);                        /* 或 ACFUN_THREADS=4 */
acfun_conv2d_int8(input, N, H, W, Cin, wq, wscale, bias,
                  3, 3, Cout, 1, 1, s_in, output, act);
/* K=27 (Cin=3, 3x3) 自动走 tap 解码消除后的多线程直接内核; 小图工作感知回落串行。 */
```
```bash
./build/acfun_bench build/model.acfun        # 全基准 (benchmark_result.txt 同步刷新)
# 隔离正确性回归:
g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
    tools/test_int8_direct.cpp build/libacfun.a -fopenmp -o /tmp/test_int8_direct
/tmp/test_int8_direct                        # 期望 ALL PASS
# 稳定 A/B 微基准 (对比基线库后 30 轮中位数):
g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
    tools/bench_int8_direct.cpp build/libacfun.a -fopenmp -o /tmp/bench_int8_direct
/tmp/bench_int8_direct
```

## 5. 工程与文档

- `src/conv_int8_direct_avx512.cpp`：tap 字节偏移预计算 + 内部像素无边界判断快速路径 +
  a4 解码提到 cb 循环外；SIMD 数值内核与 epilogue 逐字未改。
- `tools/bench_int8_direct.cpp`：新增内核级 30 轮中位数微基准（A/B 稳定加速口径）。
- `benchmark_result.txt`：全基准实测刷新 + v2.0.0 稳定 A/B 对比表。
- `README.md`：同步特性、路线与目录说明。

## 6. 下一步

- 移动端 **ARM NEON** 后端（GEMM / 卷积 / 量化 / 池化）—— 当前最大差距；
- 更细的**图优化**（算子重排 / 常量折叠 / Conv→Pool 跨层流式融合以彻底消去中间张量）；
- 把本版「内部像素无边界判断」范式推广至 Winograd 变换域与 INT8 Winograd 输出变换；
- INT8 微内核 ILP 与更大 KC 的进一步压榨（当前 INT8 大矩阵受内存带宽饱和限制）。
