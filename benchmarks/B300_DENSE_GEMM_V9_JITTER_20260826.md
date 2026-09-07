# B300 `dense_gemm_v9` 抖动问题记录（2026-08-26）

## 结论

当前观察到的“抖动”包含两个不同层次，不能混为一个指标：

1. **kernel 内部的 pipeline 交接抖动**：TMA producer 的 `tma_wait_empty` 与 MMA
   consumer 的 `mma_wait_ab_full` 在同一个 K-loop 中反复出现。producer 和 consumer
   的总完成时间接近，但瞬时进度呈追赶、阻塞、再追赶的 burst，而不是平滑稳态。
2. **未插桩 benchmark 的端到端延迟抖动**：同一轮 50 个 CUDA-event sample 的
   min、median、p95 和 max 差距明显，不同 Modal run 的绝对性能也有变化。现有结果
   没有保存逐样本顺序，也没有同步记录每个 sample 的 SM clock、power 和温度，因此
   只能确认存在宏观波动，不能把它完全归因于功耗降频。

这两个问题会相互叠加：pipeline burst 直接降低单个 cluster 的工作效率；GPU
功耗/频率状态则改变整个 launch 的绝对时间，并可能掩盖小幅代码收益。

> 状态更新：完成 AB6 profile、恢复 AB5 后，kernel 源码现已切换为
> `AB_STAGES=4`，并完成一次未插桩 B300 复测。本文原有细粒度数值来自 AB6
> profile，并同时引用 AB5/AB6 对比结果；它们是历史测量事实，不随当前常量回写。

> 状态更新（2026-08-26）：加入 cluster 级 Z-order 栅格化后，`mma_wait_ab_full`
> 抖动的直接主因从“交接 burst”进一步定位到“hbm 带宽近饱和下的完成延迟方差”。
> Z-order 把波长从 64×2 变为约 16×10，AB6 同口径下 wait 总量 -29.7%、长尾 -47%、
> 未插桩单样本 custom/TE 从 76.44% 升至 98.39%。见第六节。

## 当前配置与测量口径

- GPU：NVIDIA B300 SXM6 AC，SM103，148 SM。
- GEMM：`M=N=K=16384`。
- 输入：rowwise MXFP8 E4M3，E8M0 scale，每 32 个连续 K 元素一个 scale。
- 累加与输出：FP32 accumulator，BF16 output，fast accumulation disabled。
- kernel：`cutex/kernels/dense_gemm_v9.py`。
- tile：2 CTA 合作完成 `256x256x128`，每 CTA 对应 `128x256x128`。
- 被记录的 profile：`AB_STAGES=6`，`ACC_STAGES=1`；当前源码为
  `AB_STAGES=4`，`ACC_STAGES=1`。
- launch：8192 CTA，即 4096 个 2-CTA cluster；每个 cluster 只处理一个 output tile。
- 未插桩性能：5 秒 GPU warmup、10 次 kernel warmup、50 个 CUDA-event sample。
- IKET：独立单次 launch，只由中部 cluster `(32,32,0)` 写入用户 ranges；全网格
  warp lifetime 仍被记录。

## 一、pipeline 交接抖动

### 1.1 现象

当前保留的 AB6 profile 关键统计如下：

| 指标 | AB6 结果 |
|---|---:|
| leader MMA `wait_ab_full` 总量 | 20.256 us / 128 次 |
| `wait_ab_full` 扣除 64 ns/次观测 floor 后的 excess | 12.064 us |
| `wait_ab_full` p95 / max | 0.448 / 0.800 us |
| `wait_ab_full > 128 ns` | 38 / 128 次 |
| CTA 64 TMA `wait_empty` excess | 20.160 us |
| CTA 65 TMA `wait_empty` excess | 21.728 us |
| sampled TMA main | 55.232 us |
| sampled MMA main | 59.936 us |
| MMA accumulator/epilogue tail wait | 5.120 us |
| median cluster span | 59.264 us |
| IKET whole-launch span | 3317.504 us |

`tma_wait_empty` 和 `mma_wait_ab_full` 的语义相反：

- `tma_wait_empty`：producer 要复用某个 ring slot，但前一代 UMMA 尚未完成读取，
  即 consumer 对 producer 施加 backpressure。
- `mma_wait_ab_full`：consumer 已走到下一个 K-stage，但 A、B、SFA、SFB 的 TMA
  transaction 尚未全部完成，即 producer/data arrival 使 consumer starvation。

如果 pipeline 已进入平滑稳态，通常应长期由同一侧成为吞吐上限；现在两种 wait
在同一个 mainloop 中反复出现，说明两侧瞬时领先关系不断翻转。

### 1.2 当前交接过程

每个 K-stage 覆盖 `K=128`，内部包含：

```text
producer:
  wait slot empty
  arm one full barrier
  enqueue TMA A
  enqueue TMA B
  enqueue TMA SFA
  enqueue TMA SFB

consumer:
  wait the same full barrier
  copy SFA/SFB from SMEM to TMEM
  issue 4 × K=32 UMMA
  commit and release the slot
```

full barrier 只有在四路 transaction 全部完成后才会放行。交接 credit 的粒度是整个
`K=128` stage，而不是单个 `K=32` UMMA，也不是四路数据中的某一路。因此任意一路
较晚到达都会延迟整个 stage；consumer 又会在一个 burst 中连续发四次 UMMA，等到
slot release 后 producer 才能重新取得对应 credit。

IKET 显示两个 producer 通常领先 MMA 3--4 个 stage，数据从最后一侧 TMA enqueue
到 MMA wait 入口的中位年龄约为 1.600 us。六级 ring 提供了 lookahead，但没有让
交接变成均匀流水。

### 1.3 AB5 到 AB6 的证据

只将 `AB_STAGES` 从 5 增至 6 后：

| 指标 | AB5 | AB6 | 变化 |
|---|---:|---:|---:|
| MMA `wait_ab_full` excess | 14.560 us | 12.064 us | -17.14% |
| CTA 64 TMA `wait_empty` excess | 22.336 us | 20.160 us | -9.74% |
| CTA 65 TMA `wait_empty` excess | 23.616 us | 21.728 us | -7.99% |
| sampled MMA main | 60.096 us | 59.936 us | -0.27% |
| median cluster span | 60.256 us | 59.264 us | -1.65% |

第六级确实增加了 TMA lookahead，并缩小了目标 wait，但大部分收益没有落到 MMA
main 和端到端性能上。`mma_s2t`、stage release、accumulator/epilogue tail 等其他
部分抵消了收益。这说明 ring 深度偏紧是问题的一部分，但不是唯一根因。

相同 `K=128` stage 不能继续直接增加到 7 级：AB6 tensor payload 已达
205824 B/CTA；AB7 仅 payload 就需要 240128 B，超过 B300 每 SM 233472 B shared
memory 上限。

### 1.4 AB4 未插桩复测

将当前源码减少为 `AB_STAGES=4` 后，在同一测试口径下完成一次 B300 复测：

- Modal run：`ap-PdPaY2euc4q5t4VDjBkvwK`；
- 编译、有限值检查和正确性均通过；
- custom 与 TE 逐元素一致，`max_abs_vs_te=0.0`；
- 本轮没有启用 IKET，不产生新的 profile。

| 配置 | custom median | custom TFLOP/s | TE median | custom / TE |
|---|---:|---:|---:|---:|
| AB6 profile 同轮基准 | 4250.352 us | 2069.498 | 3249.008 us | 76.44% |
| **AB4 当前复测** | **4214.240 us** | **2087.231** | **3245.536 us** | **77.01%** |

AB4 相对 AB6 的 custom median 快 0.85%，而同轮 TE median 只变化 -0.11%；归一化
比例提高 0.57 个百分点。不过 AB4 自身 50 个 sample 的 min/max 为
3458.272/4253.152 us，`(max-min)/median=18.86%`。因此这次结果可以确认 AB4 没有
造成明显退化，但 0.85% 尚未超过当前测量抖动，不能判定为真实 stage 收益。

AB4 payload 为 137216 B/CTA；两个 CTA 合计 274432 B，仍超过每 SM 233472 B，
所以 stage 从 5/6 降到 4 并没有把 occupancy 从一 CTA/SM 提升到两 CTA/SM。没有
新的 IKET 数据，也不能据此判断 `mma_wait_ab_full` 是否按预期增大。

### 1.5 对全 kernel 的放大

shared-memory 占用使每个 SM 只能驻留一个 v9 CTA。148 个 SM 最多同时运行 74 个
2-SM cluster：

```text
4096 cluster / 74 resident cluster = 约 56 waves
```

当前不是 persistent kernel。每一波都重新执行 prologue、128-stage mainloop、
epilogue、TMEM 释放和 CTA tail barrier。因此 cluster 内数微秒级的等待和尾部开销
会在约 56 波中反复出现在关键路径上，不能由同一 resident worker 的下一块 tile
覆盖。

## 二、端到端延迟抖动

### 2.1 当前 AB6 同轮分布

50 个未插桩 CUDA-event sample 的聚合统计：

| 实现 | min | median | p95 | max | `(max-min)/median` |
|---|---:|---:|---:|---:|---:|
| `dense_gemm_v9` | 3280.544 us | 4250.352 us | 4432.096 us | 4443.040 us | 27.35% |
| TE/cuBLASLt | 3189.600 us | 3249.008 us | 3700.992 us | 3707.648 us | 15.94% |

当前 benchmark 先连续测 custom kernel，再连续测 TE，并非交错测量。CUDA events
排除了 Python launch overhead，但不能排除这两个连续区间经历了不同的 GPU
power/clock 状态。

结果文件只保留 mean、median、min、max 和 p95，没有保存 50 个 `samples_us`，因此
当前无法回答：

- 快样本是否全部集中在测量开头；
- 延迟是单调漂移、周期振荡，还是两个稳定状态之间跳变；
- custom 和 TE 是否在相同瞬时 SM clock/power 下运行。

### 2.2 不同正式 run 之间的变化

| 配置 | custom median | TE median | custom / TE |
|---|---:|---:|---:|
| AB5 run 1 | 4054.464 us | 3139.328 us | 77.43% |
| AB5 run 2 | 4231.232 us | 3304.768 us | 78.10% |
| AB6 保留 run | 4250.352 us | 3249.008 us | 76.44% |
| AB4 当前复测 | 4214.240 us | 3245.536 us | 77.01% |

绝对延迟在不同 run 间变化，但 custom/TE 的相对比例仍处于约 76%--78%。这支持
“公共 GPU 状态影响绝对值”的解释，同时也说明它不足以解释 v9 相对 cuBLASLt 的
稳定性能差距。

AB6 相对 AB5 的差值与上述运行间波动处在同一量级，因此目前不能仅凭端到端时间
判断 AB6 比 AB5 更快或更慢。可确认的变化是 IKET 中 full/empty wait 指标下降，
但尚未转化为超出噪声范围的未插桩性能收益。

## 三、当前证据能够与不能够说明什么

### 已确认

1. TMA producer 与 MMA consumer 的总进度接近，不是某一侧从头到尾持续慢很多。
2. `tma_wait_empty` 与 `mma_wait_ab_full` 都存在显著长尾，pipeline 交接呈 burst。
3. AB6 增加了 lookahead，并降低两类 wait，但未显著降低 MMA main 或端到端延迟。
4. `ACC_STAGES=1` 使 5.120 us accumulator/epilogue tail 位于 tile 的关键路径。
5. 一 CTA/SM、2-SM cluster 和约 56 waves 的执行模型放大了每 tile 的等待与尾部。
6. 未插桩延迟存在明显的 sample 内和 run 间波动。
7. **动态定位到主因（2026-08-26 Z-order 节）**：行主序发射下，每波是
   全 M 列 × 1–2 N 列的瘦长足迹，把 A/B 面板反复重读，使 whole-launch 读出约
   15 GB、等效约 65% HBM 带宽，把 TMA 完成延迟放大到 1.2–2.6 us、且呈随机；
   换用 Z-order 栅格化后波长变为约 16 × 10，读出降为约 6.6 GB，`mma_wait_ab_full`
   总量从 20.256 us 降到 14.240 us（AB6 口径，`-29.7%`）。

### 尚未确认

1. Z-order 后剩余 wait 长尾的直接原因仍含 HBM/L2 miss 与 TMA engine 排队两种可能；
   需要 counter-based profile 区分。
2. TMA、Tensor Core、L2、HBM 的实际利用率和 backpressure counter。
3. 50 个 benchmark sample 的时间顺序，以及对应的 power、clock、temperature。
4. cuBLASLt 本次选中 kernel 的 tile、cluster、stage、persistent 和 epilogue 策略。

## 四、测量限制

- IKET range 最低观测值约为 64 ns，包含 event 和 wait 指令本身；本文的 `excess`
  只是每次扣除 64 ns 后的敏感性统计，不等于无插桩硬件 stall 的精确时长。
- `tma_issue` 和 `mma_issue` range 只覆盖异步指令发射，不能当作 TMA 搬运时间或
  Tensor Core 实际执行时间。
- 当前用户 ranges 只来自一个中部 cluster。全网格 warp lifetime 可以用于计算
  launch span 和 cluster span，但不能提供其他 cluster 的细粒度 stage 信息。
- IKET trace 不能提供 cuBLASLt proprietary kernel 的内部 pipeline 对照。
- 当前未插桩结果没有保留逐样本数组，限制了对宏观抖动形态的回溯分析。

## 五、后续验证要求（不包含优化）

如果要继续定位而不是立即改代码，最小验证集合是：

1. 未插桩 benchmark 保留每个 `samples_us` 及顺序，并同步采样 SM clock、power、
   temperature 和 throttling reason。
2. custom 与 TE 采用交错或分组交错测量，避免两个实现分别落在不同 GPU 状态区间。
3. 只抓一次 counter-based profile，检查 Tensor Core active、TMA/SMEM stall、L2 hit、
   DRAM throughput；如果 B300 counter 工具仍失败，应明确记录工具边界，不用 IKET
   range 推导不存在的硬件 counter。
4. 对 TE 单次 launch 读取 cuBLASLt algorithm metadata 和 grid/block/cluster 信息，
   用于区分算法工作划分差异与单纯 power/clock 差异。

## 六、Z-order 栅格化修复（2026-08-26）

针对 1.2 节确认的“交接呈 burst”现象，2026-08-26 在 `dense_gemm_v9.py` 中加入
cluster 级 Z-order（Morton）栅格化：kernel 仍读物理 `block_idx`，但把
`cluster_linear`（行主序 cid）经 `_zorder_decode` 解码成 64 × 64 cluster tile 网格上的
tile 坐标，供 TMA 切片和 mma 切片使用。grid、`(2,1,1)` cluster、tile、
warp 角色和 full/empty mbarrier 协议均不变；同一 cluster 的两 CTA 共享同一个
cid，因此配对与 `mma_tile_coord_v` 语义不变。

### 机制

行主序发射下，任意约 74-cluster 波都是“全 M × 1–2 N”的瘦长足迹，把 A/B 面板
反复重读。以 `warpLifetimes` 重建的 AB6 基线波中位单元为 277 MB、whole-launch
15.25 GB；Z-order 让连续 cid 在 tile 空间聚成约 16 × 10 的方形波，波中位 119 MB、
whole-launch 6.57 GB（约 2.3×）。HBM 压力下降后 TMA 完成延迟的均值与方差都减小，
`mma_wait_ab_full` 因而收敛。

### AB6 同口径对比（唯一变量 = 栅格化）

| 指标 | AB6 线性（基线） | AB6+Zorder | 变化 |
|---|---:|---:|---:|
| `mma_wait_ab_full` 总量 | 20.256 us | **14.240 us** | **-29.7%** |
| `mma_wait_ab_full` excess | 12.064 us | **6.016 us** | **-50.1%** |
| >128ns 长尾次数 | 38 / 128 | **20 / 128** | **-47%** |
| wait p95 | 448 ns | **192 ns** | **-57%** |
| wait max | 800 ns | 992 ns | +24%（长尾右移已由稀疏化抵消） |
| TMA 完成延迟 p50 | 1984 ns | **1168 ns** | **-41%** |
| TMA 完成延迟 max | 2624 ns | **1632 ns** | **-38%** |
| 未插桩单样本 custom / TE | 76.44% | **98.39%** | **+22 pp** |

> 单样本输入为 0 warmup / 1 iteration，绝对 μs 只用于同轮对照；IKET 为单独 launch。

### AB4+Zorder 对照

同为 Z-order、改为 `AB_STAGES=4` 时，`mma_wait_ab_full` 总量为 20.032 us、
长尾 55/128。四段 ring 比六段更紧，把剩余 1 µs 级完成延迟暴露为更多短 wait；
因此保留 `AB_STAGES=6` 才是更优组合。

### 未插桩性能

| 运行 | custom median | TE median | custom / TE | 备注 |
|---|---:|---:|---:|---|
| AB6 线性基线 | 4250.352 us | 3249.008 us | 76.44% | 50-sample |
| AB4+Zorder | 2976.032 us | 2779.264 us | 93.05% | 单样本 |
| AB6+Zorder | 2783.040 us | 2707.008 us | 98.39% | 单样本 |

端到端提升的一部分来自功耗/时钟状态（三次运行不同），但 Z-order 在 IKET 层的
wait 与 TMA 完成延迟上的下降是机制性的。要把它变成可复现的基准结论，仍需一次
正式 50-sample 未插桩测量。

## 七、问题关闭标准

只有同时满足以下条件，才认为“抖动问题”已经得到解释：

- pipeline 层：能够把主要 `mma_wait_ab_full` 长尾关联到具体资源或调度事件，而不
  只是观察到 barrier wait；
- benchmark 层：能够从逐样本时序解释快、慢区间，并给出对应的 GPU 状态证据；
- 性能比较层：custom 与 TE 在可比的瞬时 GPU 状态下测量，改动收益大于重复测量
  噪声，而不是只比较两次独立 run 的 median。

## 原始材料

- 当前源码：`cutex/kernels/dense_gemm_v9.py`（含 Z-order 栅格化，`AB_STAGES=6`）
- AB6 线性 IKET profile：
  `artifacts/20260825T175241.998683Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.trace.json`
- AB6+Zorder IKET profile：
  `artifacts/20260826T044016.789793Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.trace.json`
- AB4+Zorder IKET profile：
  `artifacts/20260826T042810.954532Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.trace.json`
- AB6 对比报告：`benchmarks/B300_DENSE_GEMM_V9_AB6_IKET_COMPARISON_20260826.md`
- 初始 IKET 分析：`benchmarks/B300_DENSE_GEMM_V9_IKET_PROFILE_20260825.md`
- 栅格化波足迹估算：`scripts/analyze_iket_v9.py`
- 机制详解：`benchmarks/B300_DENSE_GEMM_V9_HBM_BANDWIDTH_ZORDER_20260826.md`
