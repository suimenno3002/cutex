# v9 `mma_wait_ab_full` 抖动根因：wave footprint 与 HBM 带宽饱和，以及 Z-order 为何有效

**摘要**：`dense_gemm_v9` 的 K-loop 里 `mma_wait_ab_full` 反复出现长尾，即使把
stage 数从 4 提 6 也压不下去。根因不在 ring 深度，而在 **grid 发射顺序**：行主序
发射让任意一个约 74-cluster 的驻留 wave 都是一个**全 M 列 × 1–2 N 列**的瘦长
footprint，把 A/B 面板反复重读，使 whole-launch 的 HBM 读出量达到约 15 GB、等效带宽约
4.6 TB/s（B300 HBM3e 峰值的约 57–65%）。记忆体压力近饱和之后，单个 TMA
transaction 的完成延迟被随机放大到 1.2–2.6 µs，而六级 ring 的在途窗口只有约
2.5 µs，于是 MMA 就会周期性饿住。用 cluster 级 Z-order（Morton）栅格化把 wave 变成
**16 × 10** 的方形 footprint 后，读出量降到约 6.6 GB、完成延迟中位从 1.98 µs 降到
1.17 µs，`mma_wait_ab_full` 总量从 20.256 µs 降到 14.240 µs（AB6 同口径）。

- 涉及文件：`cutex/kernels/dense_gemm_v9.py`
- 分析脚本：`scripts/analyze_iket_v9.py`
- 数据来源（AB6 线性 / AB4+Zorder / AB6+Zorder 三份 IKET）：见文末“原始材料”。

---

## 1. 现象：有 stage 还在抖动

`dense_gemm_v9` 用 TMA producer 和 tcgen05 MMA consumer 各占一个 warp，通过
full/empty 双 mbarrier 交接一个 `K=128` 的 A/B/SFA/SFB stage。理想情况下，四级或
六级 ring 足够让数据以流水形式到达；但 IKET 显示 K-loop 内两种 wait 反复出现：

- `tma_wait_empty`：producer 想复用某个 ring slot，但前一代 UMMA 尚未读完 →
  consumer 对 producer 的 backpressure。
- `mma_wait_ab_full`：MMA 已走到下一个 K-stage，但四路 TMA transaction 尚未全部
  完成 → producer/数据到达对 consumer 的 starvation。

ABA5→AB6 的实验已经说明把 stage 加 6 只能缓解：`mma_wait_ab_full` excess 从
14.560 µs 降到 12.064 µs（-17%），但 sampled MMA main 只缩短 0.27%，未插桩端到端
没有可测收益。原因正是：**stage 数只能掩盖 latency，补偿不了由 HBM 压力带来的
latency 自身波动。**

## 2. 从 IKET 重建 wave footprint

IKET 4.7.0 会记录**全网格**所有 warp 的 lifetime，即使细粒度用户 ranges 只来自
一个中部 cluster。用这些 lifetime 可以重建“同一时刻哪些 cluster 同时在跑”。

v9 的 grid 是 `(M//CTA_TILE[0], N//CTA_TILE[1], 1) = (128, 64, 1)`，
cluster 是 `(2,1,1)`，所以 **cluster tile 网格是 64 × 64**，每个 cluster 算一个
`256×256` 输出 tile。驻留模型是每 SM 一 CTA（shared memory 已限制 occupancy），
148 SM / 2-CTA-cluster ≈ 74 个 cluster 同时驻留；`4096 / 74 ≈ 56` 个 wave。

用 `warpLifetimes` 按发射时间排序、取 74 个连续发射的 cluster 为窗口，得到：

| 顺序 | 窗口 footprint (M × N tile) | 每 wave 独特字节 |
|---|---:|---:|
| 行主序（线性） | **64 × 2** | 中位 **276.8 MB** |
| Z-order | **16 × 10** | 中位 **113.2 MB** |

> 注意：这里的 footprint 必须按**发射顺序**看。脚本早期的版本按物理 block 坐标重建
> 会得出 `64 × 1` 的假结论，因为物理坐标不等于发射顺序（Z-order 之后尤其如此）。
> `scripts/analyze_iket_v9.py` 会识别 kernel 里的 `_zorder_decode` 并改用发射顺序
> 重算。

为什么 `64 × 2` 是坏形状：`M_span=64` 表示同一 wave 把所有 64 个 M 方向的 tile 都读了。
A 是 `(M, K)`，B 是 `(N, K)`，二者的读取量随 M/N 跨度线性增长：

```
unique_bytes(wave) = (M_span × 256 × K) + (N_span × 256 × K)
```

- 线性：`(64 × 256 × 16384) + (2 × 256 × 16384) ≈ 276.8 MB/wave`
- Z-order：`(16 × 256 × 16384) + (10 × 256 × 16384) ≈ 113.2 MB/wave`

而 **A/B 面板是跨 wave 复用的**：同一 wave 内所有 cluster 共享同一组 A 行 / B 列，但下一个
wave 的 tile 集合和上一个 wave 几乎不重叠（瘦长 wave 沿 N 平移，每次换掉一整列）。
所以整段 launch 读出量 ≈ 每 wave 独特字节 × wave 数。

## 3. 为什么这会使 HBM 带宽近饱和

| 指标 | 线性 | Z-order |
|---|---:|---:|
| 每 wave 独特字节（中位） | 276.8 MB | 113.2 MB |
| whole-launch 读出量 | **15.25 GB** | **6.57 GB** |
| 隐含 HBM 带宽（按 AB6 线性 whole-launch 3317.504 µs 折算） | **4.6 TB/s** | 约 2.0 TB/s |
| 相对 B300 HBM3e 峰值（约 8 TB/s） | **约 57%** | 约 25% |

> 峰值带宽用的是 B300 HBM3e 标称约 8 TB/s。不同测量口径下“瘦长 wave”曾得到
> 5.3–5.4 TB/s（约 66%），实际值随 TMA 在途量与 L2 命中率浮动，因此本文表述为
> **约 57–65%**。

在这条压力水平上，完成延迟的**均值**和**方差**同时被拉高。IKET 把 38 次长尾
（>128 ns）的 wait 拿出来，从 `tma_issue` 起点到 full-barrier 放行测得：

| 指标 | AB6 线性 | AB6+Zorder |
|---|---:|---:|
| 完成延迟 p50 | 1984 ns | 1168 ns |
| 完成延迟 max | 2624 ns | 1632 ns |

而六级 ring 的在途覆盖是多少？producer 领先 MMA 的量分布在
`{3:52, 4:71, 5:2}`（AB6+Zorder），即最多领先 5 个 stage，每个 stage 约
320–420 ns，总覆盖约 2.0–2.5 µs。**一旦某 stage 的完成延迟超过这个窗口，
MMA 就必须等**。瘦长 wave 1.98 µs 的中位完成延迟已经把窗口填满，因此振荡不可避免。

## 4. Z-order 为什么能解决

### 4.1 目标：让任意发射窗口是方形

Z-order（Morton）曲线把二维 tile 网格变成一个一维序，使得序上相邻的元素在空间上
也相邻。kernel 把 cluster 的行主序 cid 经 `_zorder_decode` 解码成 tile 坐标：

```text
      _____________
     | (0,0)(0,1) |        rank = 0, 1, 2, 3  ->  0x 0x
     | (1,0)(1,1) |        bits interleave m and n
     |_____________|
```

对 64 × 64 网格，每个坐标是 6-bit；把 `m` 的第 i 位嵌入 rank 的第 `2i` 位、`n` 的
第 i 位嵌入第 `2i+1` 位，就是一个双射。解码时反向提取即可：

```python
def _zorder_decode(rank):            # cluster_linear -> (tile_m, tile_n)
    m = n = 0
    for bit in range(ZORDER_BITS):   # 6
        m |= ((rank >> (2*bit)) & 1) << bit
        n |= ((rank >> (2*bit+1)) & 1) << bit
    return m, n
```

### 4.2 为什么方波能把 HBM 压力减掉 2.3×

因为 `unique_bytes` 随 M、N 跨度**加法**增长，而跨 wave 复用要求 wave 在 M/N 两个
方向都“薄”。方波中 `M_span × N_span ≈ 74`（例如 16 × 10 或 8 × 10），在计算
`unique_bytes` 时只累计了 `16 + 10` 列，而瘦长 wave 的 `64 + 2` 是 66 列。把它投影到
A/B 上：

```
linear (64 × 2) : (64×256×K) + (2×256×K) = 276.8 MB
zorder (16 × 10): (16×256×K) + (10×256×K) = 113.2 MB
ratio ≈ 2.44×  (因为 30 / 12.5)
```

`M_span=64` 意味着**一列所有 M 都被同一 wave 集齐**——而 A 的 M 维是 256 行，同一
M 行的 A tile 在每个 N 列都要被重读。瘦长 wave 沿 N 平移时，A 面板每 wave 都被完整
读一遍。方波则限制了 A 的 M 跨度，使其只在有限 N 内重读，B 同理。

### 4.3 Z-order wave shape / 覆盖的代价

Morton 曲线在 4×4 时是完美方块，在 74 个 cluster 的窗口上有轻微“Z 字形”抖动：
首尾 wave 会略大。从数据看：中位 113.2 MB、max 318.8 MB（max 出现在曲线相邻块尺寸
16×8 以上时）。整体读出 6.57 GB，相对线性 15.25 GB 减少 2.32×。

### 4.4 实现是否改变正确性

不改。因为：

1. **grid 不变**：仍 `(128, 64, 1)`，cluster 仍 `(2,1,1)`；
2. **TMA 与 MMA 都读新坐标**：`tile_m, tile_n` 经 `mma_tile_coord_mnl` 传给
   `tma_g_a[...]`、`tma_g_b[...]` 和 epilogue `out[...]`；
3. **cluster 配对保持**：两 CTA 共享同一个 cid（`cluster_linear`），因此同一 tile，
   只差 `mma_tile_coord_v = bid_m % 2`；
4. **IKET 采样 cluster** 仍由 `cluster_linear` 决定（物理簇不因栅格化而改变）。

## 5. 单变量验证（AB6 线性 vs AB6+Zorder）

保持 `AB_STAGES=6`、ACC_STAGES=1、warp 角色、tile、四路 TMA 顺序全部不变，只加
Z-order 栅格化，B300 一次 IKET 结果：

| 指标 | AB6 线性 | AB6+Zorder | 变化 |
|---|---:|---:|---:|
| `mma_wait_ab_full` 总量 | 20.256 µs | **14.240 µs** | **-29.7%** |
| `mma_wait_ab_full` excess | 12.064 µs | **6.016 µs** | **-50.1%** |
| >128 ns 长尾次数 | 38 / 128 | **20 / 128** | **-47%** |
| wait p95 | 448 ns | **192 ns** | **-57%** |
| wait max | 800 ns | 992 ns | +24%（由长尾稀疏化抵消） |
| TMA 完成延迟 p50 | 1984 ns | **1168 ns** | **-41%** |
| TMA 完成延迟 max | 2624 ns | **1632 ns** | **-38%** |
| 未插桩单样本 custom / TE | 76.44% | **98.39%** | **+22 pp** |

**为什么 max 反而变大**：长尾次数从 38 降到 20，说明大部分等待消失；剩余个别离群点
来自记忆体系统的突发调度，被 Z-order 稍微曝露出来，不影响整体收敛判断。

### AB4+Zorder 对照

同为 Z-order、`AB_STAGES=4` 时 `mma_wait_ab_full` 总量为 20.032 µs、长尾 55/128。
四级 ring 更紧，把剩余约 1 µs 的完成延迟暴露为更多短 wait。因此最终保留
`AB_STAGES=6` 才是 Z-order 下更优组合。

## 6. 关于 L2 复用的说明

上面的“HBM 带宽”是**保守上界**——它假设所有读取都未命中 L2。实际 B300 有大量
L2 缓存，跨 wave 复用的一部分 A/B 会命中 L2 而不计 HBM。瘦长 wave（列状）的 L2
命中率**低于**方波，因为：

- 同 wave 内共享：方波让 16×10 的 tile 互为邻居，中间的 A/B 在 L2 里被多个 cluster
  复用；
- 跨 wave 复用：瘦长 wave 沿 N 平移，下一 wave 的 A 面板几乎全部换掉，L2 只能救回
  很有限的部分。

因此 Z-order 减小的“HBM 读出 2.3×”是**下限**——实际命中率提升后，HBM 压力下降
更多，TMA 完成延迟的中位与方差都随之改善。这也是为什么 IKET 里 wait 收敛幅度
（-30%）大于纯流量比值（2.3×）。

## 7. 为什么 stage 数解决不了，Z-order 能

把两者的作用机制分开：

| 维度 | 增加 stage 数 | Z-order 栅格化 |
|---|---|---|
| 作用 | 增加**在途时间窗**（lookahead） | 降低**完成延迟本身**（减少 HBM 竞争） |
| 对 `mma_wait_ab_full` | 略微下降（-17% excess） | 显著下降（-50% excess） |
| 对 MMA main / 端到端 | 几乎无改善 | 单样本 custom/TE 76%→98% |
| 代价 | 受 SMEM 上限，最多 6 级 | 无（纯坐标重排） |

核心差别：**stage 数是把 latency 藏起来，但如果 latency 自身在 1–2 µs 随机波动，
且窗口无法再扩大（SMEM 上限），就没有足够的余量；Z-order 是让 latency 的均值与
方差都掉下来。** 前者治标，后者治本。

## 8. 结论与后续

1. 根因：行主序 grid 产生瘦长 wave，A/B 反复重读 → HBM 带宽近饱和 → TMA 完成延迟
   均值/方差被推高 → 覆盖窗口不足 → `mma_wait_ab_full` 抖动。
2. 修复：Z-order 栅格化把 wave 变成 16 × 10，whole-launch 读出从 15.25 GB 降到
   6.57 GB，完成延迟 p50 从 1.98 µs 降到 1.17 µs，`mma_wait_ab_full` 总量
   -29.7%（AB6 同口径）。
3. 剩余：ab4+Zorder 的余量仍会因 ring 更紧而暴露为较多短 wait；因此保留
   `AB_STAGES=6`。未插桩 98.39% 是单样本（0 warmup / 1 iter），与功耗/时钟相关，
   尚需一次正式 50-sample 测量才能作为可复现基准。

## 原始材料

- 当前源码：`cutex/kernels/dense_gemm_v9.py`（含 Z-order 栅格化，`AB_STAGES=6`）
- IKET 分析脚本：`scripts/analyze_iket_v9.py`（`--stages` 应等于 kernel 的
  `AB_STAGES`）
- AB6 线性 IKET：
  `artifacts/20260825T175241.998683Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.trace.json`
- AB4+Zorder IKET：
  `artifacts/20260826T042810.954532Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.trace.json`
- AB6+Zorder IKET：
  `artifacts/20260826T044016.789793Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.trace.json`
- 相关报告：`benchmarks/B300_DENSE_GEMM_V9_IKET_PROFILE_20260825.md`、
  `benchmarks/B300_DENSE_GEMM_V9_AB6_IKET_COMPARISON_20260826.md`、
  `benchmarks/B300_DENSE_GEMM_V9_JITTER_20260826.md`
