# B300 dense_gemm_v9 IKET profile 报告（2026-08-25）

## 结论

`dense_gemm_v9` 的结果与同一组预量化输入上的 Transformer Engine
`general_gemm`（cuBLASLt-backed）逐元素一致，因此当前性能差距不是 shape、
数据格式或数值合同不一致导致的。

两轮 5 秒 GPU warmup、10 次 kernel warmup、50 个 CUDA-event samples 中，
v9 达到同轮 cuBLASLt-backed TE 吞吐的 **77.43%–78.10%**；v9 的中位延迟
高 **28.03%–29.15%**。IKET 证据指向三个主要原因：

1. 五级 A/B/SFA/SFB ring 在稳态中仍同时出现 producer backpressure 和
   consumer starvation；TMA 与 MMA 总体进度接近，但逐 K-tile 交接呈突发状，
   没有形成持续无气泡的流水。
2. accumulator commit 之后还有约 **4.32 us** 的 epilogue/release 尾部，约占
   sampled MMA leader main range 的 **7.19%**；当前 non-persistent kernel 无法把
   这段尾部与下一 output tile 的 prologue/mainloop 重叠。
3. 每 CTA 仅 staged tensor payload 就使用 **171,520 B** shared memory；B300
   实测每 SM 为 **233,472 B**，因此 shared memory 已把 residency 限制为每 SM
   一个 v9 CTA。4096 个 2-CTA cluster 至少分成 56 个执行波次，实测 cluster
   中位 span 乘以 56 与 IKET 全 kernel span 基本吻合。

这次只增加可观测性和采集编排，没有修改 tile、stage 数、warp 角色或调度策略，
也没有实施性能优化。

## 测试口径

- GPU：NVIDIA B300 SXM6 AC，SM103，148 SM。
- CuTe DSL / IKET：`nvidia-cutlass-dsl==4.7.0`。
- GEMM：`M=N=K=16384`，rowwise MXFP8 E4M3，E8M0 scale，FP32 accumulate，
  BF16 output。
- custom 与 TE 共用同一批预量化 data/scales；量化不在计时区间。
- 正式性能使用未启用 IKET 的普通编译和 CUDA Events。IKET 调用在普通编译中
  被剥离；profile 是独立子进程中的单次 launch。
- 4.7.0 的 `run-iket` 尚无 `--enabled-cluster`。kernel 内以 block 坐标做
  warp-uniform predicate，只让中部 cluster `(32,32,0)` 发出用户 ranges；对应
  CTA `(64,32,0)` 和 `(65,32,0)`，位于 SM 40/41。
- `--max-ts-cnt-per-warp=2048`；共得到 26 个 range name、1519 个完整 ranges。

IKET 是 warp-level、32 ns 粒度的实验性 in-kernel tracing。异步 TMA/MMA 的
`issue` range 只表示发射侧；完成延迟应从相应 mbarrier wait 观察。细粒度 ranges
本身也有开销，因此下文不把嵌套 range 跨 warp 相加，也不把 IKET 绝对时间当成
正式性能数字。方法和时间戳语义见 NVIDIA 的
[IKET profiling guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html)。

## 正确性与未插桩性能

两次正式运行的 correctness 都是：

- custom relative L2 vs FP32 BF16-input reference：`0.0377092361`；
- custom relative L2 vs TE：`0.0`；
- custom max abs vs TE：`0.0`；
- all finite：`true`。

| run | v9 median | v9 TFLOP/s | cuBLASLt-backed TE median | TE TFLOP/s | v9 / TE | v9 延迟增幅 |
|---|---:|---:|---:|---:|---:|---:|
| `20260825T120601.584045Z` | 4054.464 us | 2169.484 | 3139.328 us | 2801.903 | 77.43% | 29.15% |
| `20260825T142805.793397Z` | 4231.232 us | 2078.849 | 3304.768 us | 2661.637 | 78.10% | 28.03% |

绝对吞吐随连续 random-input 负载下的功耗/频率状态变化，但两轮相对比例稳定。
因此功耗状态解释绝对值波动，不足以解释 v9 相对 cuBLASLt 的固定差距。

## IKET trace 完整性与全网格行为

profile 运行的用户 ranges 只来自一个中部 cluster，但 IKET 4.7.0 仍记录了全网格
49152 个 warp lifetimes（8192 CTA × 6 warp）。由这些 lifetime 计算：

| 指标 | 结果 |
|---|---:|
| IKET kernel launch span | 3367.168 us |
| cluster 数量 | 4096（64 × 64） |
| cluster span min / median / p95 / max | 47.904 / 60.256 / 62.720 / 66.528 us |
| sampled cluster span | 62.016 us |
| sampled cluster 在 launch 内的区间 | 1674.464–1736.480 us |

中部 cluster 落在 launch 中间，span 接近全网格中位数，适合作为稳态样本。
同次运行在 profile 之前测得的未插桩单样本为 3408.384 us；IKET launch span 与它
接近，但正式性能仍以上一节的 50-sample CUDA Events 为准。

## sampled cluster 时间线

下列时间以该 cluster 最早用户 range 为 `t=0`。不同 warp 的 range 会并行，不能
纵向求和。

| 角色 / range | 起点 | 终点 | duration | 含义 |
|---|---:|---:|---:|---|
| MMA leader `v9_prologue` | 0.032 us | 0.768 us | 0.736 us | barrier/TMEM/TMA partition 等公共准备 |
| MMA leader `mma_main` | 0.832 us | 60.928 us | 60.096 us | reduction、acc commit 与 tail wait |
| 两个 TMA warp `tma_main` | 0.832 us | 56.896 us | 各 56.064 us | 128 个 K-tile load 与 ring drain |
| MMA `mma_commit_acc` | 56.480 us | 56.512 us | 0.032 us | 只测 commit 发射 |
| epilogue `epi_wait_acc` | 约 1.25 us | 约 56.90 us | 55.52–55.65 us | epilogue warp 等待 accumulator ready |
| epilogue `epi_t2r` | 约 56.93 us | 约 57.73 us | 0.768–0.832 us | TMEM 到 registers，并转 BF16 |
| epilogue `epi_store` | 约 57.73 us | 约 59.97 us | 1.664–2.176 us | `CopyUniversalOp` 直接写 GMEM |
| MMA `mma_wait_acc_empty_tail` | 56.576 us | 60.896 us | 4.320 us | 等 8 个 epilogue warp 释放 accumulator |

TMA main 结束与 MMA accumulator commit 仅相差约 0.38 us，说明宏观上 producer
和 consumer 的总进度接近；问题不是某一侧从头到尾慢很多，而是细粒度交接存在
反复停顿。

## mainloop 等待

| role / wait | 128 次总时长 | 占对应 main | median | steady p95 | max | 大于 128 ns |
|---|---:|---:|---:|---:|---:|---:|
| CTA 64 TMA `wait_empty` | 30.528 us | 54.5% | 0.192 us | 0.522 us | 1.152 us | 121 / 128 |
| CTA 65 TMA `wait_empty` | 31.808 us | 56.7% | 0.192 us | 0.544 us | 1.152 us | 123 / 128 |
| leader MMA `wait_ab_full` | 22.752 us | 37.9% | 0.096 us | 0.522 us | 1.120 us | 44 / 128 |

一个无明显 stall 的 wait range 在该 trace 中最低约为 64 ns，其中包含两端 IKET
event 与 wait 指令本身。仅作为敏感性检查，扣除每次 64 ns floor 后，仍分别有
22.336 us、23.616 us 和 14.560 us 的“高于观测 floor 的等待相关时间”。这不是
无插桩硬件 stall 的精确值，但足以证明 0.3–1.1 us 的长尾并非只由 32 ns timer
量化产生。

语义上：

- `tma_wait_empty` 表示 producer 想重用第 5 级 ring slot 时，前一代 MMA 尚未完成
  对该 slot 的读取，属于 consumer 对 producer 的 backpressure。
- `mma_wait_ab_full` 表示 MMA 到达某 K-tile 时四路 TMA transaction 尚未全部完成，
  属于 producer/data arrival 对 consumer 的 starvation。
- `tma_issue` 和 `mma_issue` 的短 duration 不能当成搬运或 Tensor Core 执行时间；
  它们只覆盖异步指令发射。

两种 wait 都出现，说明当前 pipeline 更像在 burst/phase 间来回追赶，而不是一个
已经填满、只由单一稳态吞吐上限控制的流水。仅凭 IKET 不能进一步断言是 HBM、L2、
TMA engine 还是 Tensor Core 的硬件利用率先到顶；那需要另一次 counter-based
profile，不能从这些 range 伪造结论。

## residency 与 grid wave

每 CTA、每 stage 的静态 payload 是：

- A：16384 B；B：16384 B；SFA：512 B；SFB：1024 B；
- 五级合计：`(16384 + 16384 + 512 + 1024) × 5 = 171520 B`，尚未计入
  barrier/TMEM allocator metadata 和 alignment。

B300 runtime properties 显示每 SM shared memory 为 233472 B；两个 v9 CTA 的 staged
payload 已达 343040 B，因此一个 SM 无法同时 resident 两个 v9 CTA。148 个 SM 对
2-CTA cluster 最多提供 74 个并发位置：

```text
ceil(4096 clusters / 74 resident clusters) = 56 waves
60.256 us median cluster span × 56 = 3374.336 us
IKET whole-launch span                  = 3367.168 us
```

两者只差约 0.21%。这说明 v9 的 end-to-end 时间确实由“一 SM 一 CTA、2-SM cluster、
约 56 波”模型主导。当前代码又是 regular non-persistent grid，每个 cluster 只做一个
256×256 output tile，所以每一波都完整支付 prologue、mainloop、epilogue 和 dealloc，
不存在同一 resident cluster 继续领取下一 tile 的机会。

## 为什么低于 cuBLASLt-backed TE

按证据强度排序：

1. **高置信：manual ring 的交接存在稳态气泡。** TMA empty wait 与 MMA full wait
   都有反复长尾，而两侧最终结束时间又接近；这是 phase granularity / handoff
   不够平滑的直接证据。
2. **高置信：epilogue/release 位于关键尾部。** accumulator commit 后 MMA leader
   还等待 4.32 us，且 non-persistent 调度不能用下一 tile 覆盖它。
3. **高置信：shared-memory residency 把执行固定成约 56 波。** cluster span × wave
   count 复现了全 launch 时间，v9 没有额外 residency 来隐藏某个 cluster 的 stall。
4. **中等置信：warp 资源有明显静态空闲。** peer CTA 的 MMA warp 只执行约
   0.448 us `mma_main`，随后在 CTA tail sync 等约 59.424 us；8 个 epilogue warps
   在 reduction 期间各等 accumulator 约 55.5 us。这是 2CTA/warp-specialized
   设计的结果，但在一 CTA/SM 的资源约束下没有别的 CTA 工作可填这些槽位。
5. **待 counter 验证：regular grid 的 L2 reuse/rasterization，以及直接 SIMT store。**
   源码没有 persistent tile scheduler、显式 rasterization 或 TMA-store epilogue；
   它们是与高度调优库实现相比的合理差异点，但 IKET range 本身不提供 cache hit、
   DRAM bandwidth 或 library 内部算法信息，因此本报告不把它们写成已证实根因。

cuBLASLt 的已编译内部 kernel 没有本仓库的 CuTe IKET markers，不能在本次 trace 中
逐阶段对照。这里的可靠比较是同轮 CUDA-event 延迟；对 cuBLASLt 内部 tile、
persistent 策略或 cache 行为的描述若没有额外 disassembly/counters，都只能是推测。

## 原始产物状态

本报告正文保留 AB5 profile 的已汇总数据。完成 AB6 对比后，AB5 原始 IKET 文件
已按“只保留当前一份 profile”的要求删除；其单样本 `result.json` 仍保留用于确认
当时的配置和正确性。当前原始 profile 及 AB5/AB6 对比见
`B300_DENSE_GEMM_V9_AB6_IKET_COMPARISON_20260826.md`。
