# B300 dense_gemm_v1 CUDA Core 基线

## 实现

`cutex/kernels/dense_gemm_v1.py` 保留正式 MXFP8 kernel 的五指针接口和固定 `M=N=K=16384` 合同，但完全不使用 Tensor Core：

- 一个 CTA 计算一个 `16 × 16` 输出 tile，共 256 threads，每个线程负责一个输出元素。
- K 方向以 16 个元素分块。
- 每轮由 256 个线程分别加载一个 A 和一个 B 元素。
- 按 cuBLAS/Transformer Engine 的 128×4 tiled scale 布局定位 E8M0 scale。
- E4M3 与 E8M0 在进入 shared memory 前手工转换、相乘为 FP32。
- 每个线程使用普通标量 FP32 乘加遍历 shared-memory tile。
- 累加结果转换为 BF16 后写回。
- 源码不调用 `cute.gemm`、MMA 或 `tcgen05`。

这是用于理解分块、协作搬运、同步和累加的数据通路，不以性能为目标。它故意没有向量化 load/store、双缓冲、异步 copy、寄存器分块或线程级多输出。

## B300 验证

环境：Modal `NVIDIA B300 SXM6 AC`、SM103、CuTeDSL 4.7.0。验证命令：

```powershell
uv run modal run modal_dense_gemm.py `
  --implementation cuda_core_v1 `
  --gpu-warmup-seconds 0 --warmup 0 --iterations 1
```

| 指标 | 结果 |
|---|---:|
| shape | 16384³ |
| 输出 | BF16 |
| all finite | PASS |
| relative L2 vs FP32 | 3.7709% |
| relative L2 vs TE | 0.00753% |
| max abs vs TE | 4.0 |
| 单发延迟 | 2346.909 ms |
| 吞吐 | 3.748 TFLOPS |

TE 对照单发为 2.883 ms / 3051.18 TFLOPS。两者的目标不同：v1 证明 CUDA Core 标量数据通路正确，正式 `dense_gemm.py` 才负责用 SM103 block-scaled Tensor Core 追求性能。

## 产物

- [实现](../cutex/kernels/dense_gemm_v1.py)
- [原始结果](../artifacts/20260823T080149.127555Z-dense_gemm_mxfp8_16384_cuda_core_v1/result.json)
- [Modal run](https://modal.com/apps/pengjixian9/main/ap-4lF2GAcb7FNUEnmp3PWpI5)

