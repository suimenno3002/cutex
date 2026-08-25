# B300 dense_gemm_v9 `AB_STAGES=6` IKET 对比（2026-08-26）

## 结论

只把 `dense_gemm_v9` 的 `AB_STAGES` 从 5 改为 6 后，完整 B300 编译、正确性、
50-sample 未插桩性能测试和一次 IKET profile 均通过。第六级 buffer 的确增加了
TMA 对 MMA 的领先量，并压低了 `mma_wait_ab_full`，但改善不足以转化为可测的
端到端性能收益：本轮未插桩中位延迟为 **4250.352 us**，没有快于五级版本的
两轮正式结果 **4054.464 us / 4231.232 us**。

因此本次结果支持“当前五级 ring 深度偏紧”，但也证明单纯增加一级不是主要性能
解法。它减少了部分 mainloop starvation，同时被其余 mainloop 开销、accumulator /
epilogue tail 和运行间功耗频率波动抵消。

## 唯一变量与测试口径

- 唯一代码变量：`AB_STAGES: 5 -> 6`；K-stage 仍为 128，四路 TMA 顺序、
  warp 角色、MMA 和 epilogue 均未改变。
- GPU：NVIDIA B300 SXM6 AC，SM103。
- GEMM：`M=N=K=16384`，rowwise MXFP8 E4M3/E8M0，FP32 accumulate，BF16 output。
- 未插桩性能：5 秒 GPU warmup、10 次 kernel warmup、50 个 CUDA-event samples。
- profile：上述 benchmark 完成后，由独立 `run-iket` 子进程单次 launch；仍只由
  cluster `(32,32,0)` 发出用户 ranges，`--max-ts-cnt-per-warp=2048`。
- Modal run：`ap-U3lC0jkMV68Rodbbv8qjrg`。

## 正确性与未插桩性能

六级版本：

- custom relative L2 vs FP32 reference：`0.0377092361`；
- custom relative L2 vs TE：`0.0`；
- custom max abs vs TE：`0.0`；
- all finite：`true`。

| 配置 | custom median | custom TFLOP/s | TE median | custom / TE | custom 延迟增幅 |
|---|---:|---:|---:|---:|---:|
| AB5 正式 run 1 | 4054.464 us | 2169.484 | 3139.328 us | 77.43% | 29.15% |
| AB5 正式 run 2 | 4231.232 us | 2078.849 | 3304.768 us | 78.10% | 28.03% |
| **AB6 本轮** | **4250.352 us** | **2069.498** | **3249.008 us** | **76.44%** | **30.82%** |

AB6 相对较接近的 AB5 run 2 慢 `0.45%`，相对更快的 AB5 run 1 慢 `4.83%`。
不同 Modal run 的功耗和时钟状态并未锁定，因此不能把这个差值归因于 stage 数；
可靠结论是本轮没有观察到未插桩性能提升。

## IKET 对比

下面 AB5 与 AB6 都来自一次完整 16384³ IKET launch。每个 wait range 的观测
floor 仍约为 64 ns；`excess` 表示每次扣除 64 ns 后的敏感性统计，而非无插桩硬件
stall 的精确值。

| 指标 | AB5 | AB6 | 变化 |
|---|---:|---:|---:|
| MMA `wait_ab_full` 总量 | 22.752 us | 20.256 us | -10.97% |
| MMA `wait_ab_full` excess | 14.560 us | 12.064 us | **-17.14%** |
| MMA wait >128 ns | 44 / 128 | 38 / 128 | -6 |
| MMA wait p95 | 0.480 us | 0.448 us | -6.67% |
| MMA wait max | 1.120 us | 0.800 us | -28.57% |
| CTA64 TMA `wait_empty` excess | 22.336 us | 20.160 us | -9.74% |
| CTA65 TMA `wait_empty` excess | 23.616 us | 21.728 us | -7.99% |
| sampled TMA main | 56.064 us | 55.232 us | -1.48% |
| sampled MMA main | 60.096 us | 59.936 us | **-0.27%** |
| median cluster span | 60.256 us | 59.264 us | -1.65% |
| IKET whole-launch span | 3367.168 us | 3317.504 us | -1.47% |
| MMA `wait_acc_empty_tail` | 4.320 us | 5.120 us | +18.52% |

第六级按照预期增加了 lookahead：

- AB5：MMA 进入 full wait 时，两个 TMA producer 已完成 enqueue 的共同领先量主要
  是 2–3 个 K-stage（125 / 128 次）；
- AB6：共同领先量主要变成 3–4 个 K-stage（124 / 128 次）；
- 数据从最后一侧 TMA enqueue 到 MMA wait 入口的中位年龄从 1280 ns 增至
  1600 ns，正好多出约一个 stage 周期。

这说明六级 buffer 的机制确实生效。问题在于收益没有完整落到 MMA main：
`mma_k_tile` 合计缩短 1.728 us，其中 `mma_wait_ab_full` 缩短 2.496 us；但
`mma_s2t`、release range 的观测时间及 mainloop 末尾的 accumulator/epilogue tail
合计抵消了大部分收益。最终 sampled MMA main 只缩短 0.160 us。

## 判断

`AB_STAGES=6` 是对目标指标有效、对端到端性能中性的变更：它缓解但没有消除
burst。由于六级 payload 已使用 205824 B/CTA，七级仅 payload 就会达到 240128 B，
超过 B300 的 233472 B/SM，不能继续用相同 K=128 stage 简单加深。

如果继续优化，下一步需要改变交接粒度或隐藏 tail，而不是再增加同尺寸 stage。
本报告只记录本轮结果，不实施下一项优化。

## 当前原始产物

- `artifacts/20260825T175241.998683Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/result.json`
- `artifacts/20260825T175241.998683Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.pftrace`
- `artifacts/20260825T175241.998683Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/iket_pid_0x10.trace.json`
- `artifacts/20260825T175241.998683Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/iket/run-iket.log`
