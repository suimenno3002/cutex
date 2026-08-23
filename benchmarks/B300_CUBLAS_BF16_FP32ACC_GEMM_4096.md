# B300 cuBLAS BF16/FP32-acc GEMM 4096³

## 结果

在 Modal `NVIDIA B300 SXM6 AC` 上直接调用 `cublasGemmEx` 测量 (M=N=K=4096)：

| 输入模式 | 中位延迟 | 持续 TFLOPS | 2250 TFLOPS 达成率 | 平均功率 | 平均 SM 时钟 | `sw_power_cap` |
|---|---:|---:|---:|---:|---:|---:|
| random uniform `[-1, 1]` | **100.383 µs** | **1369.17** | **60.85%** | 992.4 W | 1506.6 MHz | 100% |
| all-one | **72.811 µs** | **1887.61** | **83.89%** | 723.7 W | 2032.0 MHz | 0% |

每次 GEMM 的逻辑运算量为：

\[
2MNK=2\times4096^3=137{,}438{,}953{,}472\ \text{FLOPs}.
\]

训练数据近似的 random 输入持续约 **1.369 PFLOPS**。低切换的 all-one 控制组保持最高 SM 时钟，但只达到约 **1.888 PFLOPS**；这说明 4096³ 除 random 输入下的功耗降频外，还存在 shape、tile/wave、pipeline amortization 或 cuBLAS 算法效率造成的约 16% 峰值差距。

## 精度和 API

- A/B/C：`CUDA_R_16BF`。
- Compute/accumulation：`CUBLAS_COMPUTE_32F`。
- API：直接 `cublasGemmEx`。
- Algorithm：`CUBLAS_GEMM_DEFAULT_TENSOR_OP`。
- cuBLAS version：`130600`。

量化或框架开销不存在；CUDA Events 只测连续 GEMM。

## 测量方法

- Phase 顺序：`random → one → one → random`。
- 每个 phase 前空闲 5 秒。
- 每个 phase 执行 5000 次 warmup。
- 之后测量 5 个 repeat，每个 repeat 连续执行 5000 次 GEMM。
- 结果取 5 个 repeat-average CUDA-event latency 的中位数。
- 功率、SM 时钟、GPU utilization、温度和 power-cap reason 以请求的 20 ms 间隔采集。

重复 phase：

| Phase | 输入 | 延迟 | TFLOPS | 功率 | SM 时钟 | power cap | 最高温度 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | random | 99.879 µs | 1376.06 | 995.5 W | 1497.8 MHz | 100% | 57 °C |
| 3 | random | 100.888 µs | 1362.29 | 989.2 W | 1515.4 MHz | 100% | 58 °C |
| 1 | one | 72.812 µs | 1887.59 | 721.2 W | 2032.0 MHz | 0% | 51 °C |
| 2 | one | 72.810 µs | 1887.64 | 726.2 W | 2032.0 MHz | 0% | 51 °C |

所有 phase 的 GPU utilization 约为 100%，`hw_thermal_slowdown` 始终为 `Not Active`。

## 与 16384³ 对比

| Shape | random TFLOPS | all-one TFLOPS | random 峰值达成率 | all-one 峰值达成率 |
|---:|---:|---:|---:|---:|
| 4096³ | 1369.17 | 1887.61 | 60.85% | 83.89% |
| 16384³ | 1490.08 | 2239.07 | 66.23% | 99.51% |

4096³ 的 random 性能比 16384³ 低约 8.1%，all-one 低约 15.7%。因此 4096³ 不是该 cuBLAS BF16 kernel 的满峰值 shape。

## 正确性

独立 CUDA FP32 reduction 验证 `C[0,0]`：

| 输入 | cuBLAS BF16 输出 | FP32 reference | 相对误差 |
|---|---:|---:|---:|
| random | 17 | 16.996683 | 0.0195% |
| one | 4096 | 4096 | 0 |

四个 phase 均为 PASS。

## 复现

```powershell
uv run modal run modal_cublas_bf16_gemm.py `
  --size 4096 --warmup 5000 `
  --iterations 5000 --repeats 5 `
  --sample-interval-ms 20 --idle-seconds 5
```

- 原始 artifact：[`../artifacts/20260822T154845.461606Z-cublas_bf16_fp32acc_gemm_4096/result.json`](../artifacts/20260822T154845.461606Z-cublas_bf16_fp32acc_gemm_4096/result.json)
- Modal run：<https://modal.com/apps/pengjixian9/main/ap-pZpVXbfvbQMXimUKEXVPde>
