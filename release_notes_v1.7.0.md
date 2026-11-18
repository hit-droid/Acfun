# Acfun v1.7.0 发行说明 — 小 K INT8 直接卷积内核（修复 0.2×–0.3× 回退，224²×3 首次反超 float）

> 本版在 v1.6.0（INT8 卷积激活量化向量化，量化遍 47–77×）之上，攻下全基准**最后一片成片回退点**：
> INT8 卷积在「输入通道极小 / stride>1」等 **K=KH·KW·Cin 很小而 M 很大**的场景慢于 float 卷积
> （`224²×3→16` 仅 0.2×、`56²×16 k1 s2` 0.3×、`28²×64 k1` 0.3×）。这类场景中 `im2col` 把输入
> 放大 KH·KW 倍、`packA` 再放大 4 倍，两段固定开销远超点积本身——正是 v1.6.0 发行说明「下一步」
> 明确点名的「极小 Cin 扁矩阵路径」。本版新增**小 K 直接卷积内核**，直接在 NHWC 量化输入上按像素
> 算点积，跳过 im2col 与 packA，`224²×3→16` 从 10.39 ms 降到 ~2.1 ms（0.2× → **1.1×，首次反超
> float**），`56²×16 k1 s2` 从 0.18 ms 降到 ~0.067 ms（0.3× → **0.9×**）。
>
> 这是 v1.4.0 修复分组卷积 0.2× 回退之后，第二次以「专用直接内核」消除成片回退的路径。所有改动
> 数值与原子路径**逐元素严格等价（max|Δ|=0）**，隔离回归跨 8 组边界全过，全基准「正确性：全部通过」。

> 基底说明：v1.7.0 基于 v1.6.0（`8cb56f5`，已与 `origin/main` 同步，无分叉）。本版仅新增小 K
> 直接卷积路径并在 `conv_int8.cpp` 增加派发；Winograd / 原 im2col+GEMM / epilogue / 量化遍 /
> 各 float 算子一律零改动、零回归。

## 1. 为什么做（v1.7.0 的动机）

INT8 卷积既有路径为「量化 → im2col(量化) → packA → INT8 GEMM + epilogue」。v1.5.0 / v1.6.0
把两段逐元素前后处理（反量化、量化）SIMD 化后，**唯一剩下的结构性瓶颈**是 im2col 与 packA
本身：

```cpp
/* v1.6.0 及以前: conv_int8.cpp — 小 K 场景的成本倒挂 */
Aq.resize(M * K);                                  // M=OH*OW, K=KH*KW*Cin
im2col_u8(inQ.data(), ..., Aq.data());             // 3x3: 输入放大 9 倍 + memset/拷贝
acfun_gemm_int8_epi(M, Cout, K, Aq.data(), ...);   // packA 再放大 4 倍
```

以 `224²×3→16`（K=27, M=50176）为例：inQ 仅 224·224·3≈150 KB，im2col 展开成
50176·27≈1.35 MB（**9×**），packA 再复写一遍 4× 布局；而真实点积仅
2·M·Cout·K≈43 M 次 MAC。两段访存固定开销压倒点积，使 INT8 反而慢于 float 卷积（0.2×）。
`28²×64 1x1`（K=64）虽走零拷贝 GEMM、无 im2col，但 packA + 窄 GEMM 仍使其只有 0.3×。

## 2. 本版做了什么

### 2.1 小 K 直接卷积内核（AVX-512 VNNI + 标量）

沿用本仓库「SIMD 后端独立编译单元 + 运行时派发」范式，新增三个编译单元/调度件：

- `src/conv_int8_direct_avx512.cpp`（`-mavx512f -mfma -mavx512vnni -ffp-contract=off`）：
  - **权重预打包**：把 `wq[K,Cout]` 一次重排为 VNNI「4-k × 16 通道」扩展布局
    `Bp[cb,kg,lane,t]=wq[(kg·4+t)·Cout + cb·16+lane]`，跨全部 M 个输出像素复用（预打包成本
    O(K·Cout)，相对 M·K·Cout 可忽略）；
  - **逐像素 dpbusd**：`vpdpbusd` 的一个 64B zmm 有 16 个 32-bit lane，lane ℓ 累加
    `Σ_{t=0..3} A[4ℓ+t]·B[4ℓ+t]`。同一像素的 4 个连续 k 输入值对全部 Cout 通道共享，故把 A 的
    16 个 lane 全置为这 4 字节（`_mm512_broadcast_i32x4`），B 的 lane ℓ 承载第 `cb·16+ℓ` 通道
    的 4 个 k 权重——**一条 dpbusd 即算完 16 通道 × 4 个 k**，无 gather、无 im2col、无 packA；
  - **tap 查表**：k→(ic, kh-pad, kw-pad) 在函数级预解码成小数组，消除最内层每像素每 k 组
    3 次整数除法（`k/Cin`、`/KW`、`%KW`）——这是 224²×3 场景的头号标量瓶颈；
  - **融合 epilogue**：`acc(int32) → float ×alpha[j] + beta[j]`（显式 mul+add 非 FMA，与标量
    同舍入）+ 无分支 `vmax` ReLU，按 16 通道掩码写回。尾通道用 mask load/store。
- `src/conv_int8_direct.cpp`：运行时按 `avx512vnni` 派发到向量后端，否则标量兜底（可移植 +
  正确性基线）。标量版逐 tap 累加、padding 以 zp=128 参与，数值与向量后端一致。

### 2.2 路由与零回归设计（`conv_int8.cpp`）

- 直接内核只在 `K ≤ 64 && !(1x1/s1/p0)` 时启用：`1x1/s1/p0` 已是零拷贝 GEMM（无 im2col 收益），
  排除以免逐像素开销反导致回退（实测 `28²×64 k1` 曾从 0.40→0.71 ms，排除后恢复 0.40 ms）；
- 3x3/s1/p1 且通道足够大的场景已被既有 Winograd 快路径截获，不进直接内核；大 K 仍走原 GEMM；
- alpha/beta 计算原样复用（直接内核与 GEMM epilogue 共用），padding 填 zp=128 的语义严格保留。

### 2.3 隔离回归测试（`tools/test_int8_direct.cpp`）

新增独立正确性测试：把量化输入经 im2col 后喂给严格标量 GEMM epilogue 作参考，与直接内核逐元素
比对。覆盖 1x1、1x1 stride2、3x3+padding、非整除 Cout、奇数尺寸、batch>1、有/无 ReLU 共 8 组，
全部 `max|Δ|=0`。编译运行方式见文件头注释。

## 3. 实测（沙箱 AMD EPYC 9K65 / AVX-512 VNNI，4 核 cgroup 配额；正确性门禁全过）

### 3.1 INT8 卷积：回退点转正（单线程，逐元素精度门 OK）

| 配置 | v1.6.0 INT8 | v1.7.0 INT8 | float 参考 | 加速 (FP/INT8) |
|------|-------------|-------------|-----------|----------------|
| `224²×3→16 k3 s1` | 10.39 ms (0.2×) | **~2.10 ms** | ~2.38 ms | **1.1×**（首次反超） |
| `56²×16→32 k1 s2` | 0.182 ms (0.3×) | **~0.067 ms** | ~0.059 ms | **0.9×** |
| `28²×64→64 k1 s1` | 0.400 ms (0.3×) | 0.399 ms（排除，零回归） | ~0.115 ms | 0.3× |

> 墙钟绝对值在共享 cgroup 主机上有 ±20% 波动，`224²×3` 区间约 2.1–4.7 ms；但方向稳定、两次
> 独立运行一致：跳过 im2col+packA 后由 0.2× 级回退稳定转为 ≥1× 正收益（最优运行反超 float）。
> `28²×64 k1` 保持原路径，数值与性能均零回归。

### 3.2 其余路径（零回归）

全基准「正确性：全部通过」，`FAIL` 计数为 0。GEMM 吞吐 / 卷积 / Winograd F(2,3)/F(4,3) /
深度卷积 / 分组卷积 / 最大池化 / FC+ReLU / Post-GEMM Epilogue / INT8 GEMM / INT8 GEMM Epilogue /
INT8 激活量化 / INT8 Winograd / BF16 正确性与相对加速与 v1.6.0 持平。端到端模型前向（3×32×32）
~0.33 ms，与基线同量级。

## 4. 用法

API 不变，`acfun_conv2d_int8` 在满足条件时自动走直接内核：

```c
acfun_conv2d_int8(input, N, H, W, Cin, wq, wscale, bias, KH, KW, Cout, stride, pad, s_in, output, act);
/* K=KH*KW*Cin<=64 且非 1x1/s1/p0 时自动派发小 K 直接内核, 其余路径原样。 */
```
```bash
./build/acfun_bench build/model.acfun        # 全基准（含量化遍标量 vs 向量对比）
# 隔离正确性回归:
g++ -O3 -std=c++17 -Iinclude -Iinclude/acfun -Isrc \
    tools/test_int8_direct.cpp build/libacfun.a -fopenmp -o /tmp/test_int8_direct
/tmp/test_int8_direct                        # 期望 ALL PASS
```

## 5. 工程与文档

- `src/conv_int8_direct.cpp` / `_avx512.cpp`：新增，小 K 直接卷积调度器 + AVX-512 VNNI 内核 + 标量兜底。
- `src/internal.h`：新增 `acfun_conv2d_int8_direct` 及两后端声明。
- `src/conv_int8.cpp`：alpha/beta 计算前移；新增小 K 路由（排除 1x1/s1/p0）。
- `CMakeLists.txt`：新增两个源文件及 AVX-512 VNNI 编译标志（含 `-ffp-contract=off`）。
- `tools/test_int8_direct.cpp`：新增隔离正确性回归测试。
- `README.md` / `benchmark_result.txt`：同步更新特性、路线与基准（全基准正确性通过）。

## 6. 下一步

- 移动端 **ARM NEON** 后端（GEMM / 卷积 / 量化 / 池化）；
- 小 K 直接卷积的**多线程并行**（当前单线程，基准场景已足够，可向大 batch 生产路径扩展）；
- 更细的**图优化**（算子重排 / 常量折叠 / Conv→Pool 跨层流式融合以彻底消去中间张量）。
