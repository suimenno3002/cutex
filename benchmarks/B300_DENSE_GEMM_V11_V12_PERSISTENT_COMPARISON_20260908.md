# B300 `dense_gemm_v11/v12` Persistent Kernel 性能对比（2026-09-08）

## 结论

`dense_gemm_v12.py` 已在 NVIDIA B300 SXM6 AC 上完成 JIT、完整
`16384³` 计算、正确性检查和正式计时。v12 确实是 occupancy-sized
persistent kernel：运行时查询得到 74 个常驻 `(2,1,1)` cluster，以
`(2,1,74)` 的物理 grid 覆盖 4096 个逻辑 output cluster，每个常驻 cluster
连续执行 55 或 56 个 `256x256` output tile。

“消除 prologue/epilogue”在这里指稳态消除，而不是删除必需的数据搬运：

- barrier 初始化和 512-column TMEM 分配从“每个 output tile 一次”变为“每个
  常驻 cluster 一次”；
- 六级 A/B/SF ring 的 stage/phase 跨 output tile 连续推进，只在整个
  persistent loop 末尾做一次 drain；
- 两份错位 accumulator view 交替使用。epilogue 完成 TMEM->RMEM copy 并执行
  `fence_view_async_tmem_load()` 后，先释放 accumulator，再做 BF16 转换和 GMEM
  store，使下一 tile 的 MMA 能与当前 tile 的后半段 epilogue 重叠；
- 第一个 tile 仍有不可避免的启动 prologue，最后一个 tile 仍有不可避免的
  drain/store epilogue，输入 load 和输出 store 也不可能被字面删除。

三轮、每轮 200 个 CUDA-event 样本的跨运行中位数如下：

| 实现 | Median latency (us) | TFLOP/s | 同轮 cuBLASLt 归一化效率 |
|---|---:|---:|---:|
| v11 non-persistent | **3562.416** | **2469.137** | **95.044%** |
| v12 persistent | **3619.824** | **2429.978** | **92.812%** |
| cuBLASLt（六轮整体中位） | **3364.152** | **2614.659** | 100% |

当前 v12 相对 v11 的跨运行中位延迟增加 **1.61%**，吞吐降低 **1.59%**。
因此本次结果证明了 persistent 执行、稳态 prologue 摊销、epilogue overlap 和
数值正确性，但**没有证明性能提升**。不能用 v12 最快一轮的 3530.304 us
选择性地宣称提速；另外两轮分别为 3619.824 和 3649.408 us。

## v11 与 v12 的结构差异

| 项目 | v11 | v12 |
|---|---|---|
| 逻辑工作 | 4096 个 2-CTA output cluster | 相同 |
| 物理 grid | 8192 CTAs；每个 CTA pair 做一个 tile 后退出 | 148 CTAs = 74 个常驻 cluster |
| Tile scheduler | grid 坐标经 CuTe Layout `8x8` block swizzle | 原生 `StaticPersistentTileScheduler`，相同 swizzle |
| 每个物理 cluster 的 tile 数 | 1 | 55 或 56 |
| A/B/SF ring | 6 stages，每个 CTA 独立建环并 drain | 6 stages，phase 跨 tile 连续，仅最终 drain |
| Barrier/TMEM 生命周期 | 每个 tile 初始化、分配和释放 | 每个常驻 cluster 只做一次 |
| Accumulator | 单个 256-column view | 两个 view：`[0,256)`、`[208,464)`，重叠 48 columns |
| Scale-factor TMEM | `[256,272)` + `[272,304)` | `[464,480)` + `[480,512)` |
| MMA/epilogue | MMA 等当前 tile 完整 store/release | TMEM->RMEM fence 后 early release，转换/store 与下一 MMA 重叠 |

v12 的物理 accumulator 总覆盖为 `256 * 2 - 48 = 464` columns；再加 SFA 的
16 columns 和 SFB 的 32 columns，正好使用 512-column TMEM allocation，scale
factor 与 accumulator 没有别名。

## 测试口径

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC |
| 架构 | SM103 (`sm_103a`) |
| CuTeDSL | NVIDIA CUTLASS DSL 4.7.0 |
| PyTorch | `2.13.0a0+9186a08b2c.nv26.07` |
| Transformer Engine | `2.17.0+2e559f06` |
| GEMM | `Y[M,N] = X[M,K] @ W[N,K].T` |
| Shape | `16384 x 16384 x 16384` |
| 输入 / scale | MXFP8 E4M3FN / E8M0FNU，每 32 个连续 K 元素一个 scale |
| 累加 / 输出 | FP32 / BF16，fast accumulation disabled |
| MMA / CTA-pair tile | `256x256x32` / `256x256x128` |
| Cluster | `(2,1,1)`，192 threads/CTA |
| Timer | CUDA events |
| GPU 预热 | 10 秒 |
| Kernel 预热 | 20 次 |
| 计时样本 | 每轮 200 次 |
| 计时边界 | 预量化 raw GEMM；量化不计时 |
| cuBLAS 对照 | Transformer Engine `general_gemm` 调用的 cuBLASLt MXFP8 路径 |

这里使用 cuBLASLt MXFP8，而不是 BF16 `cublasGemmEx`，因为只有前者与 v11/v12
共享相同的预量化 E4M3 数据、E8M0 block scale、FP32 accumulator 和 BF16 输出
合同。

## 原始结果

`归一化效率 = custom TFLOP/s / 同轮 cuBLASLt TFLOP/s`。每一行的 custom 和
cuBLASLt 使用同一远端运行中的同一份预量化输入。

| 实现 | 轮次 | Custom median (us) | Custom p95 (us) | Custom TFLOP/s | cuBLASLt median (us) | cuBLASLt TFLOP/s | 归一化效率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v11 | 1 | 3567.728 | 3640.768 | 2465.461 | 3375.392 | 2605.947 | 94.609% |
| v11 | 2 | 3562.416 | 3670.272 | 2469.137 | 3385.856 | 2597.893 | 95.044% |
| v11 | 3 | 3376.896 | 3491.168 | 2604.786 | 3240.240 | 2714.642 | 95.953% |
| v12 | 1 | 3530.304 | 3710.592 | 2491.597 | 3343.328 | 2630.939 | 94.704% |
| v12 | 2 | 3619.824 | 3795.840 | 2429.978 | 3359.632 | 2618.172 | 92.812% |
| v12 | 3 | 3649.408 | 3731.168 | 2410.279 | 3368.672 | 2611.146 | 92.307% |

绝对性能有运行间波动：六轮 cuBLASLt median 为 3240.240–3385.856 us。汇总时
同时使用 run median 和同轮 cuBLASLt 归一化，不使用单次 `min` 或最优轮排序。

## 正确性

六轮正式运行以及 v12 的 `0/0/1` smoke test 全部为 `PASS`：

- 所有输出有限；
- v11/v12 与同轮 cuBLASLt 输出逐元素 BF16 一致：`relative_l2=0.0`、
  `max_abs=0.0`；
- custom 与 cuBLASLt 相对原始 BF16 输入的 FP32 `torch.mm` reference 都是
  `relative_l2=0.03770923614501953`；
- 输出 dtype 为 BF16。

这也端到端验证了 v12 的 rank-stride work 分配：74 个物理 cluster 确实完整且
不重不漏地写出了全部 4096 个逻辑 output tile。

## 如何解释负收益

已确认的事实是：v12 正确执行 persistent loop，但三轮汇总比 v11 慢 1.59%。
本次没有采集功耗、时钟、L2/HBM counter 或逐阶段 trace，下面只能作为后续分析
假设，不能当作已经证实的硬件归因：

1. 每个 output tile 有 128 个 K-stage、每 stage 四条 block-scaled MMA，当前
   `K=16384` 下 mainloop 占主导；一次性 barrier/TMEM 初始化可摊销的比例很小。
2. v12 每个 tile 增加静态 scheduler 的 work-index、swizzle 坐标解码以及循环
   控制；这些稳态开销可能抵消 prologue/tail 节省。
3. 当前 early release 在整个 CTA-local accumulator 完成 TMEM->RMEM copy 后发生。
   CUTLASS 通用实现还能按 subtile 调整读取顺序并更早释放 48-column overlap；这是
   下一步最值得隔离验证的 epilogue 优化。
4. random-normal MXFP8 16384³ 会形成高功耗持续负载，Modal B300 的不同运行存在
   时钟差异；同轮 cuBLASLt 归一化已经降低影响，但没有替代 telemetry/counter。

因此 v12 当前应作为“正确的 persistent 基线”保留，不能替换 v11 作为性能默认值。

## 复现命令

```bash
uv run modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v11 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 20 --iterations 200

uv run modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v12 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 20 --iterations 200
```

若本机设置了会切断长时间 gRPC 流的 SOCKS 代理，可以只对该命令临时移除代理
环境变量；不要修改仓库或系统代理配置。

## 原始 artifacts 与 Modal runs

- v11 round 1：
  [`artifacts/20260908T025426.875469Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json`](../artifacts/20260908T025426.875469Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json)
  (`ap-drHfNCyR6nr5Jibg9HrINg`)
- v11 round 2：
  [`artifacts/20260908T025742.019832Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json`](../artifacts/20260908T025742.019832Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json)
  (`ap-DFqeH7m56wcjyQ5DzDCODp`)
- v11 round 3：
  [`artifacts/20260908T030059.388153Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json`](../artifacts/20260908T030059.388153Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json)
  (`ap-micoWJRCk7yiNJUifzsX7I`)
- v12 round 1：
  [`artifacts/20260908T025530.460341Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json`](../artifacts/20260908T025530.460341Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json)
  (`ap-VPWgFrXWbvjl7xkg7NjrpR`)
- v12 round 2：
  [`artifacts/20260908T025643.077411Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json`](../artifacts/20260908T025643.077411Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json)
  (`ap-xobnU7N8iAgTqAEVrOaQbS`)
- v12 round 3：
  [`artifacts/20260908T025850.173355Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json`](../artifacts/20260908T025850.173355Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json)
  (`ap-fxcUwMs6XyHwgc8YwKW4cc`)
- v12 smoke（不纳入正式汇总）：
  [`artifacts/20260907T210057.315002Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json`](../artifacts/20260907T210057.315002Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json)
  (`ap-wzBVbXA7uHEcuE54j9SJIB`)
