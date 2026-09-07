# B300 `dense_gemm_v9/v10/v11` Thread Block Swizzle 性能对比（2026-08-26）

## 结论

在 NVIDIA B300 SXM6 AC 上，三个版本都通过了正确性检查。正式测试固定
`M=N=K=16384`、10 秒 GPU 预热、20 次 kernel 预热，并用 CUDA events 记录
200 个样本；输入量化不在计时区间内。

- v9（4-stage、cluster tile 行主序）中位延迟为 **4301.344 us**，
  **2044.964 TFLOP/s**，达到同轮 Transformer Engine（TE）的 **77.51%**。
- v10（6-stage、手写 Morton/Z-order）两轮 run-median 平均为
  **3614.432 us**，**2433.979 TFLOP/s**，平均达到同轮 TE 的 **92.31%**。
- v11（6-stage、CuTe Layout 8x8 cluster block swizzle）两轮 run-median
  平均为 **3539.768 us**，**2485.057 TFLOP/s**，平均达到同轮 TE 的
  **93.71%**。

v11 相对 v10 的原始平均延迟低 2.07%，但更可靠的同轮 TE 归一化收益分别为
**1.48%** 和 **1.51%**。确认轮的 TE 中位值只差 0.015%，此时 v11 的中位延迟
仍低 **1.53%**。因此，CuTe Layout block swizzle 相对手写 Morton 是一个小幅、
两轮方向一致的改进，而不是数量级变化。

v9 到 v10/v11 的差异不能全部归因于 swizzle：v9 的 `AB_STAGES=4`，v10/v11
均为 `AB_STAGES=6`。只有 v10 与 v11 保持了 mainloop、stage、tile、cluster 和
warp 角色相同，隔离比较了 rasterization 方式。

## 测试口径

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC |
| 架构 | SM103 (`sm_103a`) |
| GEMM | `Y[M,N] = X[M,K] @ W[N,K].T` |
| Shape | `16384 x 16384 x 16384` |
| 输入 | MXFP8 E4M3FN |
| Scale | E8M0FNU，每 32 个连续 K 元素一个 scale |
| 累加 / 输出 | FP32 / BF16，fast accumulation disabled |
| MMA / CTA-pair tile | `256x256x32` / `256x256x128` |
| Cluster | `(2,1,1)`，192 threads/CTA |
| Timer | CUDA events |
| GPU 预热 | 10 秒 |
| Kernel 预热 | 20 次 |
| 计时样本 | 每轮 200 次 |
| 计时边界 | 只计 GEMM kernel；量化不计时 |

v9 与 v10 首轮差异已经很大，因此 v9 只运行一轮。v10 与 v11 首轮的差异较小，
各补一轮相同口径的确认测试。

## 原始结果

`TE efficiency = TE median / custom median`，数值越高表示 custom 越接近同轮 TE。

| 实现 | 轮次 | Custom median (us) | Custom p95 (us) | TFLOP/s | TE median (us) | TE efficiency |
|---|---:|---:|---:|---:|---:|---:|
| v9 row-major, AB4 | 1 | 4301.344 | 4344.480 | 2044.964 | 3333.904 | 77.508% |
| v10 Morton, AB6 | 1 | 3659.376 | 3727.904 | 2403.714 | 3351.712 | 91.592% |
| v10 Morton, AB6 | 2 | 3569.488 | 3682.048 | 2464.245 | 3320.528 | 93.025% |
| v11 CuTe Layout S8, AB6 | 1 | 3564.544 | 3645.632 | 2467.663 | 3313.936 | 92.969% |
| v11 CuTe Layout S8, AB6 | 2 | 3514.992 | 3646.592 | 2502.450 | 3320.048 | 94.454% |

v11 第一轮出现一个很快的 `min=2764.640 us` 离群样本，因此不使用 mean/min
做版本排序；中位数与 p95 均保持在稳定区间。不同 Modal run 的绝对时钟状态仍有
变化，这也是同时保留 TE 归一化结果的原因。

## 正确性

五轮正式运行均为 `PASS`：

- 所有输出有限；
- custom 与同轮 TE 的 `relative_l2=0.0`、`max_abs=0.0`；
- custom 与 TE 相对 FP32 reference 的 `relative_l2` 都为
  `0.03770923614501953`；
- 输出 dtype 为 BF16。

v11 首次远程编译还捕获到一个 JIT 分阶段规则问题：普通 `if` 内的 `raise` 会被
视为设备侧 early exit。尺寸检查已改为 `cutlass.const_expr(...)` 编译期分支；该次
失败没有产生性能数据，修复后本地 scaffold 回归和两轮 B300 正式运行都通过。

## 比较

### v9 到 v10/v11

以 v10/v11 两轮 run-median 的平均值和 v9 单轮结果比较：

| 比较 | 延迟变化 | TFLOP/s 说明 |
|---|---:|---|
| v9 -> v10 | -15.97% | v10 两轮平均 2433.979 TFLOP/s |
| v9 -> v11 | -17.71% | v11 两轮平均 2485.057 TFLOP/s |

这里同时改变了 stage 数与发射顺序，只能说明“当前完整版本”的端到端差异，不能作为
单变量 swizzle 结论。

### v10 到 v11：隔离 rasterization

两轮分别使用各自同轮 TE 做归一化：

| 轮次 | v10 `custom/TE` 延迟比 | v11 `custom/TE` 延迟比 | v11 归一化收益 |
|---|---:|---:|---:|
| 1 | 1.0918x | 1.0756x | 1.48% |
| 2 | 1.0750x | 1.0587x | 1.51% |

确认轮尤其适合直接比较：v10/v11 的 TE median 分别为 3320.528/3320.048 us，
只差 -0.015%；custom median 分别为 3569.488/3514.992 us，v11 低 1.53%。

结论是：对当前固定 64x64 cluster-tile 网格，CuTe Layout 的 8x8 block swizzle
略优于 Morton。两者都已经消除了 v9 行主序的瘦长 wave；v11 的额外收益较小，符合
“局部 8x8 block 比 Morton 的跨块边界更规则”这一行为，但本次测试没有采集 L2/HBM
counter，因此不能把 1.5% 进一步归因到某个具体缓存计数器。

## 复现命令

```bash
uv run modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v9 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 20 --iterations 200

uv run modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v10 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 20 --iterations 200

uv run modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v11 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 20 --iterations 200
```

## 原始结果与 Modal runs

- v9：
  `artifacts/20260826T084220.183244Z-dense_gemm_mxfp8_16384_manual_pipeline_v9/result.json`
  (`ap-gTipPziqniUsXGg5AHTrdN`)
- v10 round 1：
  `artifacts/20260826T084431.533686Z-dense_gemm_mxfp8_16384_manual_pipeline_v10/result.json`
  (`ap-qI3mscisTkXFM8Ov3GyioU`)
- v11 round 1：
  `artifacts/20260826T084714.423866Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json`
  (`ap-T9ACXSvrUkfx6LbbMUoaQU`)
- v10 round 2：
  `artifacts/20260826T084957.210583Z-dense_gemm_mxfp8_16384_manual_pipeline_v10/result.json`
  (`ap-yj6mn85Tcdv5cUhCeoN0U6`)
- v11 round 2：
  `artifacts/20260826T085109.294225Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json`
  (`ap-3xR4gvzij5JkCHvAEItBgW`)
