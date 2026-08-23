# B300 cuBLAS BF16/FP32-acc 4096³ 测量协议调优

## 目标

在尽量不触发 1100 W power cap 的前提下，通过调整单发测量前的 warmup 数和样本间隔，提高 random BF16 输入的实测 TFLOPS。

被测运算始终不变：

- `M=N=K=4096`。
- A/B/C 为 BF16。
- `cublasGemmEx`，`CUBLAS_COMPUTE_32F`。
- 每次 GEMM 为 `137,438,953,472 FLOPs`。
- CUDA event 只包围一发被测 random GEMM；warmup 和间隔均不计入 latency。

## 搜索空间

执行了两轮各 48 个候选的协议搜索：

- 每样本 warmup 数：`0, 1, 2, 4, 8, 16, 32, 64`。
- 样本间隔：`20, 50, 100, 250, 500, 1000 ms`。
- 搜索阶段每个候选测 3 次。
- `nvidia-smi` 请求 5 ms 间隔采集 `sw_power_cap`、SM clock、功率和温度。
- 第一轮 warmup 使用同一份 random BF16 数据。
- 第二轮 warmup 使用低切换 all-one BF16 数据，被测 GEMM仍使用 random 数据。

## 主要结果

| 方法 | 中位延迟 | TFLOPS | power-cap 观察 | 说明 |
|---|---:|---:|---:|---|
| 1000 ms 间隔，无 warmup，10 次 | 158.928 µs | 864.79 | 0% | 每次都是深度空闲后的冷态 launch |
| 连续 5000 warmup + 持续测量 | 100.383 µs | 1369.17 | 100% | 进入持续功耗墙 |
| random warmup 网格最佳，3 次搜索 | 84.160 µs | 1633.07 | 0% | 250 ms 间隔、2 发 warmup |
| random warmup 最佳协议，10 次复测 | **85.104 µs** | **1614.95** | **0%** | 250 ms 间隔、2 发 warmup |
| all-one warmup 网格最佳，3 次搜索 | 85.088 µs | 1615.26 | 0% | 100 ms 间隔、16 发 warmup |
| 5 ms 间隔、2 发 random warmup，10 次 | 84.992 µs | 1617.08 | 0% | 仅 6 个 telemetry samples |

低切换 all-one warmup 没有超过 random warmup，说明约 1.61 PFLOPS 的短突发上限不是由 warmup 数据本身造成。继续增加 warmup 数也没有稳定收益；较多 random warmup 反而略微降低随后一发 GEMM 的性能，与短时间动态功耗开始影响频率一致。

## 推荐口径

推荐使用：

1. 每个样本前等待 250 ms。
2. 执行 2 发不计时 random GEMM warmup。
3. CUDA event 测量随后 1 发 random GEMM。
4. 重复 10 次，报告 latency 中位数。

该协议的 10 次结果：

| Sample | 延迟 | TFLOPS |
|---:|---:|---:|
| 1 | 86.432 µs | 1590.14 |
| 2 | 85.024 µs | 1616.47 |
| 3 | 85.184 µs | 1613.44 |
| 4 | 85.024 µs | 1616.47 |
| 5 | 84.992 µs | 1617.08 |
| 6 | 85.216 µs | 1612.83 |
| 7 | 85.184 µs | 1613.44 |
| 8 | 85.760 µs | 1602.60 |
| 9 | 84.960 µs | 1617.69 |
| 10 | 83.712 µs | 1641.81 |

汇总：

- 中位延迟：**85.104 µs**。
- 中位吞吐：**1614.95 TFLOPS**。
- B300 2250 TFLOPS dense BF16 roofline 达成率：**71.78%**。
- 相比 1 秒间隔、无 warmup：提升 **86.7%**。
- 相比连续 power-cap 测量：提升 **18.0%**。
- 237 个 5 ms telemetry samples 中，`sw_power_cap` Active 为 **0%**。
- SM clock telemetry：2032 MHz。
- 最高温度：37 °C；thermal slowdown 为 `Not Active`。

5 ms 间隔方案的中位数高 0.13%，但整个短测试只有 6 个 telemetry samples，无法像 250 ms 方案那样有力地支持“未观察到 power cap”。因此把 250 ms / warmup 2 的 **1.615 PFLOPS** 作为更稳健的无持续功耗墙结果。

## 结论

协议调优消除了深度空闲冷态的大部分损失，也避免了连续测量的持续 power cap，但 random 输入仍未达到 all-one 在相同 shape 下的 1887.61 TFLOPS。当前数据支持：

- 约 0.865 PFLOPS 是深度空闲后孤立 launch 的冷态结果。
- 约 1.369 PFLOPS 是连续运行、持续撞 power cap 的结果。
- 约 **1.615 PFLOPS** 是当前可重复、未观察到持续 power cap 的 random-input 短突发结果。
- 约 1.888 PFLOPS 是相同 shape、低切换 all-one 数据的上限。

5 ms 遥测仍无法排除单发约 85 µs 内部的瞬时功率限制；若要证明逐微秒功耗行为，需要 NVML/CUPTI 更高频采样或允许锁频、调 power limit 的裸机。

## 产物

- random-warmup 48 点搜索：[`../artifacts/20260822T171052.746308Z-cublas_bf16_fp32acc_gemm_4096_protocol_sweep/result.json`](../artifacts/20260822T171052.746308Z-cublas_bf16_fp32acc_gemm_4096_protocol_sweep/result.json)
- all-one-warmup 48 点搜索：[`../artifacts/20260822T171628.546947Z-cublas_bf16_fp32acc_gemm_4096_protocol_sweep_warmup-one/result.json`](../artifacts/20260822T171628.546947Z-cublas_bf16_fp32acc_gemm_4096_protocol_sweep_warmup-one/result.json)
- 推荐协议 10 次复测：[`../artifacts/20260822T171207.665155Z-cublas_bf16_fp32acc_gemm_4096_spaced_250ms_random_warmup2/result.json`](../artifacts/20260822T171207.665155Z-cublas_bf16_fp32acc_gemm_4096_spaced_250ms_random_warmup2/result.json)
- 5 ms 间隔复测：[`../artifacts/20260822T171733.926795Z-cublas_bf16_fp32acc_gemm_4096_spaced_5ms_random_warmup2-same/result.json`](../artifacts/20260822T171733.926795Z-cublas_bf16_fp32acc_gemm_4096_spaced_5ms_random_warmup2-same/result.json)
- random-warmup 搜索 Modal run：<https://modal.com/apps/pengjixian9/main/ap-WF1LNqP4N0UDAJziGlmNdb>
- all-one-warmup 搜索 Modal run：<https://modal.com/apps/pengjixian9/main/ap-odGRJtdTMLJGJWiWL4kqFa>
- 推荐协议 Modal run：<https://modal.com/apps/pengjixian9/main/ap-kI3No0PE2c9bBEj7MaJlb1>
