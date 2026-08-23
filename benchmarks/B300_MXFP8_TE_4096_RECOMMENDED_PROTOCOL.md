# B300 Transformer Engine MXFP8 4096³ 推荐协议性能

## 结论

在 Modal 的 `NVIDIA B300 SXM6 AC` 上，以“每个样本等待 250 ms、执行 2 发不计时 warmup、再用 CUDA Events 测 1 发”的推荐协议测试 10 次，rowwise-only MXFP8 Fprop 的结果为：

| 指标 | 结果 |
|---|---:|
| 中位延迟 | **49.216 µs** |
| 中位吞吐 | **2792.57 TFLOPS** |
| 4500 TFLOPS roofline 达成率 | **62.06%** |
| 最快样本 | 47.008 µs / 2923.74 TFLOPS / 64.97% |
| 最慢样本 | 67.680 µs / 2030.72 TFLOPS / 45.13% |
| 逐样本 TFLOPS 算术平均 | 2575.74 TFLOPS |
| 正确性 | PASS，relative L2 vs FP32 = 3.7706% |

该口径测的是避免持续功耗墙的短突发 raw GEMM，不是持续负载吞吐。相对仓库原先 200 次 warmup、每组连续 1000 发的正式持续基线 2441.70 TFLOPS，本次中位数高 14.37%，但两者测量口径不同，不能互相替代。

## 精度合同

| 项目 | 配置 |
|---|---|
| 运算 | `Y = X @ W.T`，Fprop |
| shape | `M=N=K=4096` |
| X / W | rowwise-only MXFP8 E4M3 |
| scale | 沿 K 每 32 个值一个 E8M0 scale，GEMM-swizzled |
| 乘法 | E4M3 × E4M3 |
| compute / accumulation | `CUBLAS_COMPUTE_32F` 合同，`use_split_accumulator=True`，关闭 fast accumulation |
| 输出 | BF16 |
| 量化 | 在计时区外 |
| FLOPs | `2MNK = 137,438,953,472` |

## 测量协议

1. 随机生成 BF16 `X` 和 `W`，在计时区外量化为 rowwise-only MXFP8。
2. 先执行一次不计时的 `general_gemm(weight_q, x_q, layout="TN")`，承担编译/初始化成本并做正确性校验；不做持续 GPU warmup。
3. 启动 5 ms `nvidia-smi` 遥测。
4. 每个样本先等待 250 ms，再执行 2 发相同随机输入的 MXFP8 GEMM warmup。
5. 用一对 CUDA Events 只包围随后 1 发 GEMM，等待结束事件完成后读取时延。
6. 重复 10 次，以单发时延的中位数为主结果。

复现命令：

```powershell
uv run modal run modal_te_mxfp8_protocol.py `
  --size 4096 --repeats 10 --launch-interval-ms 250 `
  --per-sample-warmup 2 --sample-interval-ms 5
```

环境为 CUDA 13.3、PyTorch `2.13.0a0+9186a08b2c.nv26.07`、Transformer Engine `2.17.0+2e559f06`，GPU SM clock 标称与观测中位数均为 2032 MHz，power limit 为 1100 W。

## 逐样本结果

| Sample | 延迟 (µs) | TFLOPS | roofline 达成率 |
|---:|---:|---:|---:|
| 1 | 65.440 | 2100.23 | 46.67% |
| 2 | 47.104 | 2917.78 | 64.84% |
| 3 | 47.104 | 2917.78 | 64.84% |
| 4 | 47.008 | 2923.74 | 64.97% |
| 5 | 62.208 | 2209.35 | 49.10% |
| 6 | 67.680 | 2030.72 | 45.13% |
| 7 | 51.200 | 2684.35 | 59.65% |
| 8 | 47.232 | 2909.87 | 64.66% |
| 9 | 47.200 | 2911.84 | 64.71% |
| 10 | 63.872 | 2151.79 | 47.82% |

样本分成约 47–51 µs 与 62–68 µs 两个区间。5 ms 遥测在包含等待和 warmup 的 2.508 秒窗口内取得 335 个样本，`sw_power_cap` 和 thermal slowdown 均为 0 次 Active，SM clock 始终为 2032 MHz。因此没有证据把慢样本归因于持续 power cap；47–68 µs 的单发区间远短于 5 ms，当前遥测也无法排除亚毫秒瞬态限制或其他运行时波动。

窗口内功率为 201.17–228.21 W，但这个窗口绝大部分是 250 ms 间隔，不能把该功率解释为 kernel 执行时功率。

## 产物

- [测量入口](../modal_te_mxfp8_protocol.py)
- [原始 result.json](../artifacts/20260822T172332.891371Z-te_mxfp8_rowwise_fprop_4096_spaced_250ms_random_warmup2/result.json)
- [Modal run](https://modal.com/apps/pengjixian9/main/ap-3nFzyse1rCZ0gA7BpAqqX9)

