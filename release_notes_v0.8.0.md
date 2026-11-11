# Acfun v0.8.0 发行说明

> 本版延续「推理速度比肩 MNN」目标，在 INT8 量化推理路径上完成三项收尾 + 一处关键正确性修复：
> 1. **INT8 Winograd 输出变换 512-bit 向量化**（速度重点，沿用 v0.6.0 float 路线）；
> 2. **量化/反量化融合：输入就地量化**（消除整张输入的独立量化遍与 `inQ` 中间缓冲）；
> 3. **修复 INT8 im2col 卷积的填充零点 bug**（`memset` 填 `zp=128`）；
> 4. **修复 INT8 Winograd 向量化量化的 `packus` 存储 bug**（通道 8..15 错乱——本版重写引入的精度回归根因）。
>
> INT8 Winograd 3×3 卷积现 **逼近甚至超过 float 卷积**（1.0–1.8×），相对 v0.7.0 提速 **1.8–2.8×**；
> 全基准（`acfun_bench`）正确性全部通过，`max|Δ|/范数 < 0.6%`（仅逐张量激活量化固有误差）。

## 1. INT8 Winograd 输出变换 512-bit 向量化（本版速度重点）

v0.6.0 已把 **float** Winograd 输出变换 `Y = AT·M·ATᵀ` 改为 512-bit scatter-add，卷积提速约 +78%；
但 **INT8** Winograd 的输出变换仍是 **标量 `int64` 嵌套循环**（每输出元素 16×4×4 乘加、零系数
`continue` 分支、ILP 受限），成为 INT8 卷积头号瓶颈——也是 INT8 卷积此前「普遍慢于 float」的根因。

本版对 INT8 输出变换做与 v0.6.0 float 同构的重构：4 个输出元素各持 16 路 `int64` 累加器（2 zmm），
对 16 个变换域元素按 **预计算符号表**（`AT[r][i]·AT[c][j]` ∈ {±1,0}）做 `vpsub`/`vpadd` scatter-add，
末了 `int64→float` 乘 `wso` 加 `bias`（+可选 ReLU）写回；`int64` 累加防大通道下 `int32` 溢出，
与标量版本逐位等价（符号表即标量 `if(!si) continue…` 的精确重排）。AVX-512 DQ 指令
（`vcvtuqq2ps` / `vinsertf32x8`）由新增编译标志 `-mavx512dq` 支撑（所有 avx512f 主机均具备）。

## 2. 量化/反量化融合：输入就地量化（消除整张输入的独立量化遍）

此前 `conv_int8` 先对 **整张输入** 做一遍 `lroundf` 量化（`inQ`），再由 INT8 Winograd 把 `inQ`
`memcpy` 进填充缓冲——等于 **两遍扫描输入** 且白占一块 `inQ` 中间缓冲。本版改由 INT8 Winograd
在填充缓冲时 **就地量化**（AVX-512 向量化，`round-half-away` 与 `lroundf` 逐位一致，再打包为
`uint8`），彻底省去独立量化遍与 `inQ` 缓冲，缓存行为也更好（量化与填充一次完成）。`acfun_conv2d_int8`
在 3×3/s1/p1 路由到本函数时不再单独量化输入，直接传入 `float` + `inv_s`。

## 3. 修复 INT8 im2col 卷积的填充零点 bug

`im2col_u8` 此前把填充区填 `0`（uint8），而反量化末了以 `−zp·Sw` 做零点校正的前提是「填充
`x_q = zp = 128`」——填 `0` 会导致每个填充列净留 `−zp·wq`，污染全部输出（此前带填充的 im2col
INT8 卷积绝对误差畸大，如 `224²×3` 的 `max|Δ|≈5.7`）。现统一填 **`zp = 128`**，与 float 路径
`padding = 0` 严格一致（`224²×3` 精度回升到 `max|Δ|≈0.04`）。

## 4. 修复 INT8 Winograd 向量化量化的 packus 存储 bug（本版精度回归根因）

v0.8.0 重写引入的向量化量化在把 16 个 `int32` 量化结果打包成 `uint8` 时，误用了
`_mm256_packus_epi16(ri16, ri16)` 再取低 128 位的写法：`packus` 是 **按 128-bit 通道独立** 打包的，
该写法只会把 `ri[0..7]` 重复两遍、并 **丢弃 `ri[8..15]`**，导致 **通道 8..15 写入了通道 0..7 的值**，
一半输入通道量化错乱、输出整体失真（`max|Δ|` 一度高达 ~19）。正确写法是用 128-bit
`_mm_packus_epi16` 把 `ri16` 的 **低 128 位 + 高 128 位** 拼成 16 个 `uint8`：

```cpp
__m256i ri16 = _mm512_cvtepi32_epi16(ri);                 // 低 16 int32 → 16 int16
__m128i ri8  = _mm_packus_epi16(_mm256_castsi256_si128(ri16),
                                _mm256_extracti128_si256(ri16, 1));  // 16 uint8
_mm_storeu_si128((__m128i*)(dst + c), ri8);
```

修复后经逐值比对（320 万随机样本）与全基准验证，量化结果与 `lroundf + ZP` 完全一致。

## 5. 实测（AVX-512 Zen5，4 核 cgroup 配额；INT8 卷积 vs float 卷积，同机 v0.7.0 → v0.8.0）

| 配置 | v0.7.0 INT8 | v0.8.0 INT8 | 提速 | v0.8.0 vs FP |
|------|------------|------------|------|-------------|
| 3×3 112²×16 | 224 img/s（0.7× FP） | **520 img/s** | **2.3×** | **1.0×** |
| 3×3 56²×32  | 400 img/s（1.5× FP） | **805 img/s** | **2.0×** | **1.5×** |
| 3×3 28²×64  | 435 img/s（1.0× FP） | **910 img/s** | **2.1×** | **1.8×** |
| 3×3 64²×32  | 381 img/s（0.9× FP） | **1003 img/s** | **2.6×** | **1.7×** |
| 3×3 113×97×16 | 341 img/s（0.8× FP） | **950 img/s** | **2.8×** | **1.1×** |
| 3×3 48²×64  | 423 img/s（0.9× FP） | **755 img/s** | **1.8×** | **1.4×** |
| 3×3 24²×8   | 1950 img/s（0.8× FP） | **5015 img/s** | **2.6×** | **0.8×** |

> 全部用例 `max|Δ|/范数 < 0.6%`、结论「Winograd 全部通过」。INT8 3×3 在中等以上通道负载上已 **超过**
> float；极小通道（Cin≤16 且 Cout 小）仍受逐元素量化开销与内存带宽约束（0.8×），尚未触及输入变换
> 向量化，后续可进一步收窄差距。

## 6. 工程与文档

- `src/winograd_int8.cpp`：新增 `wino_output_block_int8`（512-bit 向量化输出变换）；填充缓冲 **就地量化**
  （向量化，含修正后的 `packus` 存储）；`acfun_winograd_conv3x3_int8` 改收 `float` 输入 + `inv_s`；
  补齐 N（batch）偏移写回。新增编译标志 `-mavx512dq`。
- `src/conv_int8.cpp`：3×3 路由传 `float` 输入（跳过独立量化遍）；`im2col_u8` 填充改填 `zp=128`。
- `src/internal.h`：更新 `acfun_winograd_conv3x3_int8` 声明。
- `README.md` / `benchmark_result.txt` 同步更新 INT8 章节；`release_notes_v0.8.0.md` 本文件。
- 新增 `tools/test_int8_iso.cpp`：INT8 Winograd 多-N 隔离正确性测试（回归用）。
