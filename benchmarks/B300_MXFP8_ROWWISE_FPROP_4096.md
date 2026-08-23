# B300 MXFP8 dense GEMM 4096³ 性能上限与对齐目标

状态：**PASS**  
测量时间：2026-08-22 20:11:55（UTC+8）  
用途：作为后续 MXFP8 `dense_gemm` 在相同 shape、布局和精度合同下必须对齐的性能基线。

## 结论

| 层级 | 中位延迟 | 吞吐 | B300 dense FP8 峰值达成率 | 定义 |
|---|---:|---:|---:|---|
| B300 硬件理论上限 | **30.542 µs** | **4500 TFLOPS** | 100% | 单卡 dense FP8 roofline，作为 MXFP8 MMA 分母 |
| NVIDIA Transformer Engine 实测基线 | **56.288 µs** | **2441.70 TFLOPS** | **54.26%** | 本仓库后续 kernel 的 100% 对齐目标 |
| 90% 阶段门槛 | 62.543 µs | 2197.53 TFLOPS | 48.83% | 达到 NVIDIA 实测基线的 90% |

后续 MXFP8 `dense_gemm` 的正式验收目标是：

```text
latency <= 56.288 µs
throughput >= 2441.70 TFLOPS
hardware_peak_efficiency >= 54.26%
```

硬件理论上限用于说明绝对 roofline；它不是当前 kernel 的通过门槛。每次 GEMM 按 `2MNK` 计算，FMA 计 2 FLOPs：

```text
FLOPs = 2 × 4096 × 4096 × 4096
      = 137,438,953,472
```

本文只保留这一组 rowwise MXFP8 Fprop 数据，不使用 Dgrad、Wgrad、BF16 或包含量化开销的结果定义性能上限。

## 固定测试合同

| 项目 | 固定口径 |
|---|---|
| GPU | 单张 NVIDIA B300 SXM6 AC |
| shape | `M=N=K=4096` |
| 逻辑运算 | Fprop，`Y = X @ W.T` |
| Transformer Engine 调用 | `general_gemm(weight_q, x_q, layout="TN")` |
| 输入数据 | `X`、`W` 都是 MXFP8 E4M3 |
| 输入 scale | 每行连续 32 个值共享一个 E8M0 scale |
| 输入存储 | 两侧均为 rowwise-only；`columnwise_data` 不存在 |
| scale 布局 | 量化阶段已生成 cuBLAS GEMM-ready swizzled scales |
| Tensor Core MMA 乘法 | FP8 E4M3 × E4M3 |
| compute / scale descriptor | `CUBLAS_COMPUTE_32F` / `CUDA_R_32F` |
| 累加合同 | FP32；`use_split_accumulator=True`，显式关闭 FP8 fast accumulation |
| 输出 | BF16 |
| 计时范围 | 仅预量化后的 raw GEMM；不含 BF16→MXFP8 量化 |
| 计时器 | CUDA Events |

运行结果断言了两个量化输入的实际存储状态：

```text
X:      rowwise=true, columnwise=false, GEMM-swizzled-scales=true
Weight: rowwise=true, columnwise=false, GEMM-swizzled-scales=true
```

MXFP8 使用 E4M3 数据和每 32 个值一个 E8M0 scale；Transformer Engine 的 `MXFP8Quantizer` 支持分别控制 rowwise/columnwise 物化。相关定义见 [NVIDIA Transformer Engine MXFP8 文档](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html) 和 [`MXFP8Quantizer` 源码](https://github.com/NVIDIA/TransformerEngine/blob/v2.17/transformer_engine/pytorch/tensor/mxfp8_tensor.py)。

累加口径以库接口合同为准：Transformer Engine 创建 `CUBLAS_COMPUTE_32F`、`CUDA_R_32F` 的 cuBLASLt matmul descriptor；`use_split_accumulator=True` 会把 `CUBLASLT_MATMUL_DESC_FAST_ACCUM` 设为 0。参见 [Transformer Engine cuBLASLt GEMM 实现](https://github.com/NVIDIA/TransformerEngine/blob/v2.17/transformer_engine/common/gemm/cublaslt_gemm.cu) 和 [cuBLASLt 文档](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmuldescattributes-t)。

## B300 硬件理论上限

[NVIDIA DGX B300 官方规格](https://www.nvidia.com/en-sg/data-center/dgx-b300/)给出整机 8 张 B300 的 FP8 Tensor Core 性能为 72 PFLOPS，并明确该数值是 2:4 sparse 规格、dense 为其一半。因此单卡 dense FP8 上限为：

```text
single_gpu_dense_peak
  = 72 PFLOPS / 8 GPUs / 2 sparse_factor
  = 4.5 PFLOPS
  = 4500 TFLOPS

ideal_latency
  = 137,438,953,472 FLOPs / 4.5e15 FLOP/s
  = 30.542 µs
```

NVIDIA 没有另外公布一个独立的“MXFP8 PFLOPS”数字。MXFP8 是 Blackwell 原生 FP8 E4M3 MMA 加 E8M0 block scaling，因此本文把官方 dense FP8 Tensor Core 峰值作为 MXFP8 MMA roofline；这是基于 NVIDIA 数据格式和硬件支持说明的推导。

本 GEMM 是普通 dense 矩阵乘法，没有 2:4 sparsity，不能使用单卡 9 PFLOPS sparse 数字作为分母。官方规格也没有单独给出 `use_split_accumulator=True` 时的峰值；严格 FP32 累加引入的开销计入实际 kernel 与 4500 TFLOPS roofline 的差距。

## NVIDIA 实现实测性能

| GPU | 操作 | shape (M×N×K) | 中位延迟 | 吞吐 | 正确性 |
|---|---|---:|---:|---:|---|
| NVIDIA B300 SXM6 AC | Fprop，`Y = X @ W.T` | 4096×4096×4096 | **56.288 µs** | **2441.70 TFLOPS（2.442 PFLOP/s）** | PASS，relative L2 vs FP32 = 3.771% |

硬件峰值达成率为：

```text
peak_efficiency
  = measured_TFLOPS / single_gpu_dense_peak_TFLOPS
  = 2441.70 / 4500
  = 54.26%

equivalently
  = ideal_latency / measured_latency
  = 30.542 / 56.288
  = 54.26%
```

这里的 54.26% 是“有效 GEMM FLOPs / 标称 dense FP8 Tensor Core 峰值”，不是直接读取的 Tensor Core pipe-active 指标。Nsight Compute 2026.2.1 已识别 B300 GB110 和实际 `nvjet_sm103_...` cuBLASLt kernel，但 Modal 当前环境的 counter measurement library 返回 `Unknown error on device 0`，所以没有可验证的 Tensor Active 百分比；不能用该指标替代或修正 54.26%。

## 测量方法与稳定性

- B300 先执行 10 秒 GPU warmup；目标 Fprop 再执行 200 次预热。
- CUDA Events 测量 10 个 repeat，每个 repeat 连续执行 1000 次 GEMM。
- 报告值是 10 个 repeat 平均延迟的中位数。
- 输入预先量化，输出 buffer 预先分配并重复使用；计时区内不分配张量。
- 固定随机种子 `20260822`，输出和 FP32 `torch.mm(X.float(), W.float().T)` 比较。

重复组平均延迟：

```text
44.728, 57.090, 56.543, 56.124, 56.307,
55.775, 56.362, 56.270, 56.339, 55.955 µs
```

首组是较高时钟下的短暂快值；其余 9 组为 55.775–57.090 µs。全 10 组中位数为 56.288 µs，去掉首组后的中位数为 56.307 µs，两者相差约 0.03%，因此基线采用未人工删样的全组中位数。

## 后续 kernel 的计算与验收

只有同时满足固定测试合同的 kernel 才能直接对齐。对任意后续实现：

```text
nvidia_alignment
  = 56.288 µs / custom_latency_us
  = custom_TFLOPS / 2441.70

hardware_peak_efficiency
  = custom_TFLOPS / 4500
  = 30.542 µs / custom_latency_us
```

| 结果 | 判定 |
|---|---|
| `nvidia_alignment >= 100%` | 达到或超过 NVIDIA Transformer Engine 实测基线 |
| `90% <= nvidia_alignment < 100%` | 阶段性接近，尚未完成正式对齐 |
| `nvidia_alignment < 90%` | 与目标差距明显，需要继续优化 |

若新 kernel 把量化、转置、scale swizzle、通信或其他 epilogue 放入计时区，应另建端到端基线，不能与本文的预量化 raw GEMM 数字直接比较。

## 环境与复现

| 项目 | 值 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC，compute capability 10.3 |
| 显存 | 275040 MiB |
| power limit | 1100 W |
| driver | 580.95.05 |
| CUDA | 13.3 |
| PyTorch | `2.13.0a0+9186a08b2c.nv26.07` |
| Transformer Engine | `2.17.0+2e559f06` |
| 计时 | CUDA Events |
| IKET | 未启用；本文只定义性能 roofline 与对齐目标 |

复现命令：

```powershell
uv run modal run modal_te_mxfp8_4096.py::main `
  --size 4096 --warmup 200 --iterations 1000 `
  --repeats 10 --gpu-warmup-seconds 10
```

原始材料：

- [benchmark 脚本](../modal_te_mxfp8_4096.py)
- [完整 result.json](../artifacts/20260822T121155.970736Z-te_mxfp8_rowwise_fprop_4096/result.json)
- [Nsight Compute 诊断](../artifacts/20260822T123022.638039Z-te_mxfp8_ncu/result.json)
- [Modal benchmark run](https://modal.com/apps/pengjixian9/main/ap-AgdVXn9OTJ6huFVMPtoNgk)
- [Modal Nsight run](https://modal.com/apps/pengjixian9/main/ap-QuvwhA53omLWAaa5fyO6vx)
- 远端 Volume：`cutex-autotune-cache/runs/NVIDIA-B300-SXM6-AC/20260822T121155.970736Z/te_mxfp8_rowwise_fprop_4096`
