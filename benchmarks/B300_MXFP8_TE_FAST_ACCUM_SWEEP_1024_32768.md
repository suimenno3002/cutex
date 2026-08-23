# B300 Transformer Engine MXFP8 fast-accum shape sweep

## 重要精度说明

这不是 FP16 accumulator 测试，因为当前 Transformer Engine 的 MXFP8 `general_gemm` 不提供 FP16 accumulator：TE 创建 cuBLASLt descriptor 时固定使用 `CUBLAS_COMPUTE_32F`，`use_split_accumulator=False` 只会设置 FP8 fast-accum 属性。Blackwell block-scaled `tcgen05.mma` 的 accumulator 也固定为 FP32。

因此，本报告测量 TE 能提供的最接近变体：

- `use_split_accumulator=False`
- FP8 fast accumulation requested
- compute contract 仍为 `CUBLAS_COMPUTE_32F`
- 输入仍为 rowwise-only MXFP8 E4M3，输出仍为 BF16

参考实现：

- [Transformer Engine cuBLASLt descriptor：compute type 固定为 32F](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu#L414-L425)
- [Transformer Engine：关闭 split accumulator 只设置 FAST_ACCUM](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/gemm/cublaslt_gemm.cu#L439-L444)
- [CUTLASS Blackwell 文档：block-scaled tcgen05 accumulator 始终为 float](https://docs.nvidia.com/cutlass/4.3.4/media/docs/cpp/blackwell_functionality.html)

## 测试口径

- GPU：NVIDIA B300 SXM6 AC（SM103，275040 MiB，1100 W power limit）
- 软件：CUDA 13.3、PyTorch `2.13.0a0+9186a08b2c.nv26.07`、Transformer Engine `2.17.0+2e559f06`
- 运算：`Y[M,N] = X[M,K] @ W[N,K].T`，且 `M=N=K`
- 计时：量化在计时区外；CUDA Events；10 秒 GPU 升频预热；每个 shape 100 次预热、500 次独立 launch，报告中位数

## 实测结果

| shape (M×N×K) | median | P95 | TFLOPS | 4500 TFLOPS 达成率 | relative L2 vs FP32 |
|---:|---:|---:|---:|---:|---:|
| 1024³ | 18.656 µs | 39.840 µs | **115.11** | 2.56% | 3.775% |
| 2048³ | 19.136 µs | 23.072 µs | **897.78** | 19.95% | 3.771% |
| 4096³ | 47.616 µs | 49.664 µs | **2886.40** | 64.14% | 3.772% |
| 8192³ | 402.432 µs | 433.408 µs | **2732.17** | 60.71% | 3.772% |
| 16384³ | 3.322 ms | 3.374 ms | **2647.68** | 58.84% | 3.772% |
| 32768³ | 26.779 ms | 27.029 ms | **2627.77** | 58.39% | 3.772% |

所有 shape 均通过有限值与 FP32 参考误差检查。误差指标与 split-accumulator sweep 相同；由于 cuBLAS 文档只承诺 FP8 fast-accum 属性影响 Ada/Hopper，不能把两次独立 B300 运行之间的性能差异解释为“FP16 累加加速”。

## 复现

```powershell
uv run modal run modal_te_mxfp8_sweep.py --fast-accum `
  --gpu-warmup-seconds 10 --warmup 100 --iterations 500
```

原始结果：`artifacts/20260822T143221.347275Z-te_mxfp8_rowwise_fprop_sweep_1024_32768_fast_accum/result.json`

