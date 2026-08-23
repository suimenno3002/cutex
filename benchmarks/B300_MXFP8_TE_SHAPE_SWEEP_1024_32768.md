# B300 Transformer Engine MXFP8 Fprop shape sweep

## 测试合同

- GPU：NVIDIA B300 SXM6 AC（SM103，275040 MiB，1100 W power limit）
- 软件：CUDA 13.3、PyTorch `2.13.0a0+9186a08b2c.nv26.07`、Transformer Engine `2.17.0+2e559f06`
- 运算：`Y[M,N] = X[M,K] @ W[N,K].T`，且 `M=N=K`
- 输入：两个 rowwise-only MXFP8 E4M3 张量，每 32 个 K 值一个 E8M0 scale
- 计算/累加：FP8 E4M3 × E4M3，FP32 accumulation contract，关闭 fast accumulation
- 输出：BF16
- 计时：量化在计时区外；CUDA Events；10 秒 GPU 升频预热；每个 shape 100 次预热、500 次独立 launch，报告中位数
- 理论参照：仓库沿用的 B300 dense FP8 Tensor Core roofline 为 4500 TFLOPS

## 实测结果

| shape (M×N×K) | median | P95 | TFLOPS | 4500 TFLOPS 达成率 | relative L2 vs FP32 |
|---:|---:|---:|---:|---:|---:|
| 1024³ | 36.736 µs | 45.472 µs | **58.46** | 1.30% | 3.775% |
| 2048³ | 36.416 µs | 37.632 µs | **471.77** | 10.48% | 3.771% |
| 4096³ | 49.328 µs | 50.624 µs | **2786.23** | 61.92% | 3.772% |
| 8192³ | 417.360 µs | 429.600 µs | **2634.44** | 58.54% | 3.772% |
| 16384³ | 3.329 ms | 3.362 ms | **2641.99** | 58.71% | 3.772% |
| 32768³ | 27.238 ms | 27.436 ms | **2583.51** | 57.41% | 3.772% |

所有 shape 均通过有限值与 FP32 参考误差检查。1024³ 和 2048³ 明显受固定 launch/调度开销限制；4096³ 达到本轮峰值 2786.23 TFLOPS；8192³ 到 32768³ 进入约 2.58–2.64 PFLOPS 的持续平台区。

## 复现

```powershell
uv run modal run modal_te_mxfp8_sweep.py `
  --gpu-warmup-seconds 10 --warmup 100 --iterations 500
```

原始结果：`artifacts/20260822T142337.268089Z-te_mxfp8_rowwise_fprop_sweep_1024_32768/result.json`

