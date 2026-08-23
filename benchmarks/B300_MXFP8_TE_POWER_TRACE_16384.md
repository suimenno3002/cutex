# B300 Transformer Engine MXFP8 16384³ 功耗与时钟追踪

## 结论

同一个 Transformer Engine rowwise MXFP8 Fprop GEMM，在相同 shape、精度合同、kernel 调用和 1100 W B300 上，仅改变输入值分布，持续性能从 **2657.29 TFLOPS** 变化到 **4479.68 TFLOPS**。

- `zero` 和 `one` 始终保持 2032 MHz，达到约 4479 TFLOPS，即单卡 4500 TFLOPS dense roofline 的 99.5%。
- `random_sign` 持续触发 `sw_power_cap`，SM 时钟降至约 1474 MHz，性能降至 3479.72 TFLOPS。
- `random` 正态输入持续触发 `sw_power_cap`，SM 时钟降至约 1125 MHz，性能降至 2657.29 TFLOPS。
- 所有模式 GPU utilization 都约为 100%；最高温度不超过 62 °C，`hw_thermal_slowdown`、`hw_slowdown` 和 `hw_power_brake_slowdown` 始终为 `Not Active`。

因此，TE 在常规随机输入下达不到标称 TFLOPS 的首要原因已经可以定位为：**输入数据切换率提高动态功耗，触发 1100 W software power cap，GPU 通过 DVFS 大幅降低 SM 时钟**。它不是 HBM 带宽上限，也不是温度降频；相同 kernel 在低切换的非零输入上能够实际达到约 4.48 PFLOPS。

## 精度与测量口径

- GPU：NVIDIA B300 SXM6 AC，SM103，275040 MiB，power limit 1100 W。
- 软件：CUDA 13.3、PyTorch `2.13.0a0+9186a08b2c.nv26.07`、Transformer Engine 2.17.0。
- Shape：`M=N=K=16384`，`Y = X @ W.T`。
- 输入：rowwise-only MXFP8 E4M3，每 32 个 K 值一个 E8M0 scale，scale 已做 GEMM swizzle。
- 乘法：FP8 E4M3 × E4M3。
- Compute/accumulation：`CUBLAS_COMPUTE_32F`，split accumulator enabled。
- 输出：BF16。
- 量化不在计时区间；GEMM 使用 CUDA events 计时。
- 单卡 dense FP8 roofline：4500 TFLOPS。

扩展实验采用 `random → zero → one → random_sign → random_sign → one → zero → random` 的对称顺序。每个 phase 前空闲 5 秒，然后依次执行 `1, 8, 32, 128, 512, 2048` 次 GEMM；`nvidia-smi` 以请求的 20 ms 间隔采集功率、SM 时钟、温度、利用率和 clock event reasons。每种输入分布独立出现两次，下面报告两次 2048-launch 持续段的均值。

## 持续结果

| 输入模式 | 输出验证 | 延迟 / GEMM | TFLOPS | 4500 达成率 | 平均功率 | 平均 SM 时钟 | `sw_power_cap` 活跃样本 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `zero` | 输出严格为 0 | 1963.553 µs | **4479.68** | **99.55%** | 996.7 W | 2032.0 MHz | 0% |
| `one` | 输出严格为 16384 | 1963.800 µs | **4479.12** | **99.54%** | 1081.2 W | 2032.0 MHz | 0% |
| `random_sign` | finite，sample max abs 544 | 2527.819 µs | **3479.72** | **77.33%** | 1094.4 W | 1473.6 MHz | 100% |
| `random` | finite，sample max abs 446 | 3310.231 µs | **2657.29** | **59.05%** | 1090.8 W | 1124.9 MHz | 100% |

`one` 是关键控制组：它产生非零结果并执行完整的 16384 项 reduction，却与 `zero` 一样达到约 4.48 PFLOPS。这排除了“全零输出被 cuBLASLt 特殊跳过”作为结果解释，支持数据相关的电路切换功耗解释。

两次持续段的 TFLOPS 分别为：

| 输入模式 | 第一次 | 第二次 | 均值 |
|---|---:|---:|---:|
| `zero` | 4479.71 | 4479.66 | 4479.68 |
| `one` | 4479.03 | 4479.21 | 4479.12 |
| `random_sign` | 3480.74 | 3478.69 | 3479.72 |
| `random` | 2667.96 | 2646.62 | 2657.29 |

重复结果稳定，且对称顺序前后的结论一致。

## 从短批次到持续状态

| 输入模式 | 8 launches | 128 launches | 512 launches | 2048 launches |
|---|---:|---:|---:|---:|
| `zero` | 4471.46 | 4479.35 | 4479.64 | 4479.68 |
| `one` | 4472.01 | 4479.01 | 4479.19 | 4479.12 |
| `random_sign` | 4333.62 | 3623.55 | 3485.19 | 3479.72 |
| `random` | 3466.15 | 2704.67 | 2669.03 | 2657.29 |

低切换输入很快达到并维持算术峰值。高切换输入在短批次后进入 software power cap：`random_sign` 降到约 3.48 PFLOPS，正态 `random` 降到约 2.66 PFLOPS。该持续时间依赖也解释了为什么单次或很短的 benchmark 会高估训练数据上的长期性能。

## 对 dense GEMM 优化目标的影响

1. 4500 TFLOPS 是这张 B300 上真实可执行的低切换数据上限，不只是纸面公式。
2. 对接近训练分布的正态随机输入，当前实例的可持续 TE 基线应使用约 **2.65–2.67 PFLOPS**，而不是 4.5 PFLOPS。
3. 自研 kernel 即使提高 Tensor Core 发射率，也可能因为功耗墙得不到相同比例的端到端收益。优化应同时减少 scale、SMEM/TMEM、barrier 和 epilogue 的无效切换，并观察功耗与频率，而不能只看 CUDA-event latency。
4. 下一层严格验证应在允许调节 power limit/锁频的裸机上重复，并使用 Nsight Compute/CUPTI 获取 Tensor pipe active 和 stall counters；当前 Modal 环境中的 Nsight Compute counter 采集仍失败。

## 复现

```powershell
uv run modal run modal_te_mxfp8_power_trace.py `
  --size 16384 --sample-interval-ms 20 `
  --idle-seconds 5 --extended-patterns
```

- 扩展控制组原始结果：[`../artifacts/20260822T145959.404245Z-te_mxfp8_power_trace_16384_patterns/result.json`](../artifacts/20260822T145959.404245Z-te_mxfp8_power_trace_16384_patterns/result.json)
- 首轮 random/zero ABBA 结果：[`../artifacts/20260822T145731.329157Z-te_mxfp8_power_trace_16384/result.json`](../artifacts/20260822T145731.329157Z-te_mxfp8_power_trace_16384/result.json)
- 扩展控制组 Modal run：<https://modal.com/apps/pengjixian9/main/ap-DqPTHslkSYrmH6vxd1elkN>
- 首轮 Modal run：<https://modal.com/apps/pengjixian9/main/ap-5CJItN6iOetMXIdQtbTTrg>

## 限制

- `nvidia-smi` 是约 20 ms 的主机侧采样，功率值可能含驱动平均窗口，不能替代片上逐周期计数器。
- 输入模式是用于因果辨别的合成数据，不代表某个具体模型层的真实 activation/weight 分布。
- 该结论适用于本次 1100 W B300 SXM6 AC 实例和当前软件栈；不同 power limit、固件或 cuBLASLt kernel 可能改变持续性能。
