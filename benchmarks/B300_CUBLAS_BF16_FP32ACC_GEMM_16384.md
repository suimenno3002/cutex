# B300 cuBLAS BF16 输入输出、FP32 累加 GEMM 基线

## 结果

在 Modal `NVIDIA B300 SXM6 AC` 上直接调用 `cublasGemmEx` 测量 (M=N=K=16384)：

| 输入模式 | 中位延迟 | 持续 TFLOPS | 2250 TFLOPS 达成率 | 平均功率 | 平均 SM 时钟 | `sw_power_cap` |
|---|---:|---:|---:|---:|---:|---:|
| random uniform `[-1, 1]` | **5.9032 ms** | **1490.08** | **66.23%** | 1078.1 W | 1277.8 MHz | 100% |
| all-one | **3.9285 ms** | **2239.07** | **99.51%** | 874.9 W | 2032.0 MHz | 0% |

每次 GEMM 的逻辑浮点运算量为：

\[
2MNK=2\times 16384^3=8{,}796{,}093{,}022{,}208\ \text{FLOPs}.
\]

random 输入是训练数据更有参考价值的持续性能：约 **1.49 PFLOPS**。all-one 是低切换功耗控制组，证明同一 API、shape 和精度合同可以达到约 **2.239 PFLOPS**，即 B300 单卡 2.25 PFLOPS dense BF16 roofline 的 99.5%。

## 精度合同

调用参数为：

```cpp
cublasGemmEx(
    handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    n, n, n,
    &alpha,
    A, CUDA_R_16BF, n,
    B, CUDA_R_16BF, n,
    &beta,
    C, CUDA_R_16BF, n,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT_TENSOR_OP);
```

- A：BF16。
- B：BF16。
- C：BF16。
- 乘法输入精度：BF16。
- compute/accumulation：FP32，`CUBLAS_COMPUTE_32F`。
- API：直接调用 cuBLAS，不经过 PyTorch `torch.mm` 或 Transformer Engine。
- cuBLAS version：`130600`。

## 测量方法

- Phase 顺序：`random → one → one → random`。
- 每个 phase 前空闲 5 秒。
- 每个 phase 先执行 100 次 warmup。
- 之后测量 5 个 repeat，每个 repeat 连续执行 200 次 GEMM。
- 每个 repeat 使用 CUDA Events 计算平均 latency；phase 结果取 5 个 repeat-average latency 的中位数。
- `nvidia-smi` 以请求的 20 ms 间隔采集功率、SM 时钟、GPU utilization、温度和 power-cap reason。
- random 与 one 分别独立出现两次，表中为两次 phase 的平均结果。

两次 random phase：

| Phase | 延迟 | TFLOPS | 平均功率 | 平均 SM 时钟 | power cap | 最高温度 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5.8845 ms | 1494.78 | 1077.8 W | 1281.0 MHz | 100% | 52 °C |
| 3 | 5.9218 ms | 1485.37 | 1078.5 W | 1274.5 MHz | 100% | 54 °C |

两次 all-one phase：

| Phase | 延迟 | TFLOPS | 平均功率 | 平均 SM 时钟 | power cap | 最高温度 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.9284 ms | 2239.08 | 876.3 W | 2032.0 MHz | 0% | 48 °C |
| 2 | 3.9285 ms | 2239.06 | 873.5 W | 2032.0 MHz | 0% | 48 °C |

所有 phase 的 GPU utilization 约为 100%，`hw_thermal_slowdown` 始终为 `Not Active`。random 输入的差距来自 1100 W software power cap 下的降频，而不是温度限制。

## 正确性

独立 CUDA reduction 用 FP32 计算 `C[0,0]` 参考值：

| 输入模式 | cuBLAS BF16 输出 | FP32 reference | 相对误差 |
|---|---:|---:|---:|
| random | -8.75 | -8.776266 | 0.2993% |
| one | 16384 | 16384 | 0 |

四个 phase 均为 PASS。

## 复现

```powershell
uv run modal run modal_cublas_bf16_gemm.py `
  --size 16384 --warmup 100 `
  --iterations 200 --repeats 5 `
  --sample-interval-ms 20 --idle-seconds 5
```

- CUDA 基准：[本仓库 `cublas_bf16_gemm.cu`](../cublas_bf16_gemm.cu)
- Modal 入口：[本仓库 `modal_cublas_bf16_gemm.py`](../modal_cublas_bf16_gemm.py)
- 原始 artifact：[`../artifacts/20260822T154510.595860Z-cublas_bf16_fp32acc_gemm_16384/result.json`](../artifacts/20260822T154510.595860Z-cublas_bf16_fp32acc_gemm_16384/result.json)
- Modal run：<https://modal.com/apps/pengjixian9/main/ap-byMQ1covbPPuEsEUpYjG6S>

## 限制

- random 输入使用确定性的 BF16 uniform `[-1, 1]`，不是某个具体模型采集到的 activation/weight。
- `nvidia-smi` 是主机侧低频遥测，功率值包含驱动采样或平均窗口；power-cap reason 比单个瞬时功率读数更适合判断是否限功耗。
- 结果适用于本次 1100 W B300、当前驱动与 cuBLAS 版本。
