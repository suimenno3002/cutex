# B300 cuBLAS BF16/FP32-acc 4096³ 单发间隔测试

## 目的

只测量 10 次独立的 4096³ random-input GEMM，每次测量前空闲 1 秒，避免持续负载触发 1100 W power cap。

- A/B/C：BF16，`CUDA_R_16BF`。
- Compute/accumulation：FP32，`CUBLAS_COMPUTE_32F`。
- API：直接 `cublasGemmEx`。
- 输入：确定性 random uniform `[-1, 1]`。
- 不做 benchmark warmup；仅在开始前执行一次不计时的 cuBLAS 初始化和正确性调用。
- 每个 CUDA-event 区间只包含一次 GEMM，1 秒等待不计入 latency。

## 10 个样本

每次 GEMM 的逻辑运算量为：

\[
2\times4096^3=137{,}438{,}953{,}472\ \text{FLOPs}.
\]

| Sample | 延迟 | TFLOPS |
|---:|---:|---:|
| 1 | 159.072 µs | 864.00 |
| 2 | 154.464 µs | 889.78 |
| 3 | 157.312 µs | 873.67 |
| 4 | 177.088 µs | 776.11 |
| 5 | 158.720 µs | 865.92 |
| 6 | 150.112 µs | 915.58 |
| 7 | 187.328 µs | 733.68 |
| 8 | 181.408 µs | 757.62 |
| 9 | 167.840 µs | 818.87 |
| 10 | 158.784 µs | 865.57 |

汇总：

| 统计量 | 结果 |
|---|---:|
| 中位延迟 | **158.928 µs** |
| 平均延迟 | 165.213 µs |
| 最小延迟 | 150.112 µs |
| 最大延迟 | 187.328 µs |
| 中位延迟对应吞吐 | **864.79 TFLOPS** |
| `sw_power_cap` Active | **0%** |

## 如何解释

这组测试达到了“不撞持续功耗墙”的目标，但它测到的是**每次空闲 1 秒后的孤立冷态 launch**，不是 warmed-up GEMM 峰值：

| 测量方式 | 中位延迟 | TFLOPS | power cap |
|---|---:|---:|---:|
| 10 次单发，每次间隔 1 秒 | 158.928 µs | 864.79 | 0% |
| 连续 warmup/持续测量 | 100.383 µs | 1369.17 | 100% |

间隔测试虽然没有功耗降频，却比连续测试慢约 36.8%。该现象与 GPU 空闲后的电源门控、HBM/L2/执行管线冷态恢复或短 kernel 无法摊销启动状态一致，但当前 20 ms `nvidia-smi` 遥测无法区分这些机制。

遥测覆盖包含 1 秒间隔的整个 wall-clock 区间，因此约 234.6 W 的平均功率主要代表空闲功率，不能解释为 100 µs GEMM 内的瞬时功率。类似地，`nvidia-smi` 报告的 2032 MHz 也不能证明每个 kernel 的前几个微秒已经处于稳定最高频率。

## 正确性

- cuBLAS BF16 `C[0,0]`：17。
- 独立 FP32 reference：16.996683。
- 相对误差：0.0195%。
- 状态：PASS。

## 复现

```powershell
uv run modal run modal_cublas_bf16_gemm.py `
  --size 4096 --warmup 0 --iterations 1 --repeats 10 `
  --input-mode random --launch-interval-ms 1000 `
  --sample-interval-ms 20 --idle-seconds 0
```

- 原始 artifact：[`../artifacts/20260822T170059.163385Z-cublas_bf16_fp32acc_gemm_4096_spaced_1000ms_random/result.json`](../artifacts/20260822T170059.163385Z-cublas_bf16_fp32acc_gemm_4096_spaced_1000ms_random/result.json)
- Modal run：<https://modal.com/apps/pengjixian9/main/ap-pdojq5CLorKDA8Ablj9IK2>
