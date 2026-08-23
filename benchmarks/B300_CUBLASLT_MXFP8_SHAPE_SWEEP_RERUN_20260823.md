# B300 cuBLASLt-backed MXFP8 shape sweep 复测

## 口径

本轮恢复最开始的持续压测方式，不使用 250 ms 间隔协议：

- GPU：Modal `NVIDIA B300 SXM6 AC`，SM103，1100 W power limit
- shape：`M=N=K`，依次测试 1024、2048、4096、8192、16384、32768
- 运算：`Y = X @ W.T`
- 输入：rowwise-only MXFP8 E4M3，沿 K 每 32 个值一个 E8M0 scale，GEMM-swizzled
- 计算：E4M3 × E4M3，`CUBLAS_COMPUTE_32F`，split accumulator 开启，fast accumulation 关闭
- 输出：BF16
- 量化：计时区外
- 预热：先用 BF16 GEMM 连续预热 GPU 10 秒；每个 shape 再执行 100 发 MXFP8 GEMM
- 计时：连续提交 500 个单发，每发由一对 CUDA Events 包围，报告 500 个时延的中位数
- 理论参照：B300 dense FP8/MXFP8 roofline 4500 TFLOPS

计时入口是 Transformer Engine 的 `general_gemm(weight_q, x_q, layout="TN")`，但实际 GEMM 后端是 `cublasLtMatmul`。TE 在本测试中负责生成 rowwise MXFP8 数据、swizzled scales 和封装调用；这不是仓库的自定义 CuTeDSL kernel，也不是绕过 TE 数据对象的独立 C++ harness。

## 2026-08-23 实测结果

| shape | median | mean | P95 | TFLOPS | 4500 TFLOPS 达成率 | relative L2 vs FP32 |
|---:|---:|---:|---:|---:|---:|---:|
| 1024³ | 17.248 µs | 18.852 µs | 19.872 µs | **124.51** | 2.77% | 3.775% |
| 2048³ | 17.136 µs | 17.623 µs | 19.008 µs | **1002.56** | 22.28% | 3.771% |
| 4096³ | 47.616 µs | 47.657 µs | 48.864 µs | **2886.40** | **64.14%** | 3.772% |
| 8192³ | 410.048 µs | 404.665 µs | 422.304 µs | **2681.42** | 59.59% | 3.772% |
| 16384³ | 3.304 ms | 3.302 ms | 3.339 ms | **2662.33** | 59.16% | 3.772% |
| 32768³ | 26.759 ms | 26.772 ms | 27.171 ms | **2629.71** | 58.44% | 3.772% |

所有 shape 均通过有限值和 FP32 参考误差检查。4096³ 是本轮最高点，为 2886.40 TFLOPS；8192³–32768³ 的持续平台为 2.63–2.68 PFLOPS，达到理论上限的 58.44%–59.59%。

## 与上一轮相同口径对比

| shape | 2026-08-22 TFLOPS | 2026-08-23 TFLOPS | 变化 |
|---:|---:|---:|---:|
| 1024³ | 58.46 | 124.51 | +112.99% |
| 2048³ | 471.77 | 1002.56 | +112.51% |
| 4096³ | 2786.23 | 2886.40 | +3.60% |
| 8192³ | 2634.44 | 2681.42 | +1.78% |
| 16384³ | 2641.99 | 2662.33 | +0.77% |
| 32768³ | 2583.51 | 2629.71 | +1.79% |

4096³ 以上两轮接近，复测提升 0.77%–3.60%。1024³ 和 2048³ 的差异超过 2 倍，说明这种“每发一对 event、由 Python 连续提交”的原始口径在很短的 kernel 上对提交队列状态高度敏感：若 GPU 已执行到 start event，而随后的 kernel 尚未进入队列，event 区间会包含这段空闲等待。因此小 shape 数字适合描述本次端到端提交状态，不适合作为稳定的纯 kernel 下界；大 shape 的持续平台更有比较意义。

## 复现

```powershell
uv run modal run modal_te_mxfp8_sweep.py `
  --gpu-warmup-seconds 10 --warmup 100 --iterations 500
```

## 产物

- [原始 result.json](../artifacts/20260823T073919.621473Z-te_mxfp8_rowwise_fprop_sweep_1024_32768_split_accum/result.json)
- [测量入口](../modal_te_mxfp8_sweep.py)
- [上一轮报告](B300_MXFP8_TE_SHAPE_SWEEP_1024_32768.md)
- [Modal run](https://modal.com/apps/pengjixian9/main/ap-9h7L5gLzGSGwbyMl5pt9eg)
- [NVIDIA Transformer Engine cuBLASLt GEMM 路径](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu)
- [NVIDIA cuBLAS 13.3 MXFP8 block scaling 文档](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-for-fp8-and-fp4-data-types)

