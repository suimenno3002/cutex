# B300 `dense_gemm_v13` CLC Persistent 优化与 cuBLASLt 对比（2026-09-09）

## 结论

`dense_gemm_v13.py` 已在 NVIDIA B300 SXM6 AC 上完成 JIT、完整
`16384³` MXFP8 GEMM、逐元素正确性检查和重复正式计时。最终版本使用
CLC dynamic persistent scheduler、M-major 8-cluster swizzle、六级连续
A/B/SF ring、两份 accumulator completion barrier 和 128×64 early-release
epilogue。

当前源码的最终 `1000` 对样本结果为：

| 实现 | Median latency | p95 | TFLOP/s | 配对效率中位数 | 更快样本 |
|---|---:|---:|---:|---:|---:|
| v13 CLC persistent | **3244.848 us** | 3296.672 us | **2710.787** | **100.229%** | **766/1000** |
| 同轮 cuBLASLt | 3252.000 us | 3306.400 us | 2704.826 | 100.000% | 232/1000（另 2 对持平） |

配对延迟差 `v13 - cuBLASLt` 的中位数为 **-7.440 us**、均值为
**-7.527 us**；逐对效率均值为 **100.232%**。同一最终配置共跑了三轮
`200` 对和两轮 `1000` 对，五轮都超过 100%。合并 2600 对后，效率中位数
为 **100.220%**、均值为 **100.225%**，`2003/2600` 对由 v13 获胜。

因此，在本文固定的 shape、数据布局、精度合同、输入和计时边界下，v13 已经
可重复地略快于同轮 cuBLASLt。领先约 0.22%，属于窄幅领先，不应外推到其他
shape、dtype、输入模式、GPU 或 CUDA/CUTLASS/Transformer Engine 版本。

## “消除 prologue 和 epilogue”的准确含义

v13 消除的是 persistent 稳态中每个 output tile 重复发生的控制开销，而不是
删除数学上必需的数据搬运：

- 每个常驻 2-CTA cluster 只初始化一次 mbarrier、分配一次 512-column TMEM；
- 六级 A/B/SFA/SFB ring 的 stage 和 phase 跨 output tile 连续推进，整个
  persistent loop 只做一次最终 drain；
- TMA descriptor prefetch、TMEM allocation 和 cluster handshake 位于持久循环
  外部；
- 两份 accumulator view 交替使用。epilogue 先搬走两份 view 重叠的 48 列，
  执行 TMEM load fence 后立即释放 empty barrier，再继续 BF16 conversion 和
  GMEM store，使下一 tile 的 MMA 与当前 tile 后半段 epilogue 重叠；
- 第一块仍有不可避免的启动 prologue，最后一块仍有不可避免的 drain；每块的
  A/B/SF load 和 C store 也必须保留。

## v12 到 v13 的结构变化

| 项目 | v12 | v13 |
|---|---|---|
| Scheduler | `StaticPersistentTileScheduler` | `ClcDynamicPersistentTileScheduler` |
| Launch | occupancy-sized `(2,1,74)`，148 CTAs | 逻辑 `(128,64,1)`，8192 CTAs；CLC 取消未驻留 cluster |
| Resident cohort | 74 个 2-CTA cluster | 同一 B300 上 74 个 2-CTA cluster |
| Work distribution | 静态 rank-stride | CLC 动态领取，尾部自动均衡 |
| Threads/CTA | 192 | 224，新增独立 scheduler warp |
| A/B/SF pipeline | 6 stages，跨 tile 连续 | 相同 payload ring，CLC response 另有单级 pipeline |
| Accumulator | 两份重叠 view，共享 empty barrier | 两份 view + 两个独立 completion barrier，共享 overlap empty barrier |
| Epilogue | 整个 CTA-local fragment 后释放 | 四个 warp 各按 128×64 subtile 排序，重叠区离开 TMEM 后提前释放 |
| Rasterization | 静态 8×8 swizzle | M-major、8-cluster CLC swizzle |
| 编译 | 默认选项 | 固定 `--opt-level 1` |

每个 CTA 的六级 tensor payload 为 `205824 B`，加手工/CLC barrier 和 TMEM
元数据后的 launch shared memory 为 `206080 B`。512-column TMEM 的物理区间为：

```text
accumulator 0: [  0, 256)
accumulator 1: [208, 464)  # 与 accumulator 0 重叠 48 列
SFA:           [464, 480)
SFB:           [480, 512)
```

## 测试合同与方法

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC |
| Compute capability | 10.3 (`sm_103a`) |
| CuTeDSL | NVIDIA CUTLASS DSL 4.7.0 |
| PyTorch | `2.13.0a0+9186a08b2c.nv26.07` |
| Transformer Engine | `2.17.0+2e559f06` |
| 运算 | `Y[M,N] = X[M,K] @ W[N,K].T` |
| Shape | `M=N=K=16384` |
| 输入 | rowwise MXFP8 E4M3FN |
| Scale | E8M0FNU，每 32 个连续 K 元素一个，tcgen05 Swizzle32x4x4 |
| 累加 / 输出 | FP32 / BF16，fast accumulation disabled |
| Custom MMA | SM103 2CTA block-scaled `256×256×32` |
| CTA-pair stage tile | `256×256×128` |
| Cluster | `(2,1,1)` |
| Timer | CUDA events |
| 正式预热 | 10 秒 GPU workload + 200 对 kernel warmup |
| 计时边界 | 预量化 raw GEMM；量化和 scale swizzle 不计时 |
| cuBLAS 对照 | Transformer Engine `general_gemm(..., use_split_accumulator=True)` 的 cuBLASLt MXFP8 路径 |

每个 sample pair 各执行一次 custom 和 cuBLASLt，奇偶 pair 交换 launch 顺序。
`配对效率 = 100 × cuBLASLt_latency / custom_latency`，所以超过 100% 表示
custom 更快。这比跨 Modal run 直接比较绝对延迟更能抵消时钟、功耗和实例状态
漂移；本文仍保留每个独立 run 的原始样本，不使用单次 `min` 或冷态结果下结论。

## 最终配置的重复性

| 轮次 | Samples | v13 median | cuBLASLt median | 配对效率 median | 配对效率 mean | v13 更快 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 3243.984 us | 3252.336 us | 100.250% | 100.257% | 167/200 |
| 2 | 200 | 3269.136 us | 3277.808 us | 100.212% | 100.212% | 151/200 |
| 3 | 200 | 3269.792 us | 3277.744 us | 100.214% | 100.157% | 145/200 |
| 4 | 1000 | 3335.456 us | 3343.520 us | 100.217% | 100.228% | 774/1000 |
| 5，当前源码 | 1000 | **3244.848 us** | **3252.000 us** | **100.229%** | **100.232%** | **766/1000** |
| 合并 | 2600 | — | — | **100.220%** | **100.225%** | **2003/2600** |

五轮绝对 latency 随 GPU 状态变化，但同轮配对效率始终同向。第五轮在修正
trace 选点与人类可读元数据之后重新构建镜像并运行，因此对应当前交付源码；
生产热路径的常量和指令路径与第四轮相同。

## 同口径 v11 / v12 / v13 对照

以下三行分别来自独立 Modal run，因此比较时优先看“同轮配对效率”，不要只看
跨 run 的绝对 latency：

| 实现 | Samples | Custom median | Custom TFLOP/s | 同轮 cuBLASLt median | 配对效率 median | Custom 更快 |
|---|---:|---:|---:|---:|---:|---:|
| v11 non-persistent | 200 | 3573.296 us | 2461.619 | 3194.496 us | 89.460% | 0/200 |
| v12 static persistent | 200 | 3392.672 us | 2592.674 | 3360.176 us | 99.002% | 3/200 |
| v13 CLC persistent | 1000 | **3244.848 us** | **2710.787** | 3252.000 us | **100.229%** | **766/1000** |

按同轮归一化效率计算，v13 相对 v12 提升 **1.239%**，相对 v11 提升
**12.038%**。v12 到 v13 的关键不是单独一项微调，而是 CLC 动态调度、独立
scheduler warp、completion barrier 分离、较小 epilogue subtile 和最终
8-cluster swizzle 的组合。

## 优化搜索与淘汰记录

先用小样本筛选，再用 200/1000 对样本验收；任何不正确、降低 residency、只在
冷态领先或正式复测低于 100% 的方案都回退。代表性结果如下：

| 变体 | Samples | 配对效率 median | 结论 |
|---|---:|---:|---|
| v12 static persistent 基线 | 200 | 99.002% | 未超过 cuBLASLt |
| CLC、M-major、swizzle 1 | 200 | 99.961% | 边缘且均值 99.827%，淘汰 |
| 每个 accumulator 独立 empty barrier | 30 | 99.239% | 回退 |
| TMA loop unroll 4 | 30 | 99.173% | 回退 |
| MMA loop unroll 4 | 30 | 100.082% | 有提升但弱于最终配置，回退 |
| branchless epilogue reverse | 30 | 100.035% | 裕量不足，回退 |
| 4 个 completion barrier | 30 | 99.863% | 回退到 2 个 |
| N-major CLC raster | 30 | 99.302% | 回退到 M-major |
| 128×32 epilogue tile | 30 | 99.106% | 回退到 128×64 |
| 48-column hybrid copy | 30 | 99.784% | 回退 |
| 全部 work/stage counter 改 Uint32 | 30 | 99.055% | 回退 |
| `(4,1,1)` cluster | 30 | 88.698% | resident clusters 从 74 降到 33，回退 |
| CuTeDSL `4.8.0.dev0` | 30 | 99.192% | 固定回 4.7.0 |
| CLC swizzle 4 | 30 | 100.163% | 候选 |
| CLC swizzle 8 | 30 | 100.216% | 进入长测并最终保留 |

还隔离测试过 AB stages、K tile、compiler/ptxas opt level、static swizzle、
TMA store、更大/更小 epilogue tile、8 epilogue warps、额外 producer warp、
不同 CLC consumer/chunk 策略、手写 scheduler 算术和多种 accumulator overlap。
它们或性能更差，或资源占用破坏 residency，或无法保持正确性，均未进入最终
源码。

一次将 SFB 的 SMEM→TMEM copy 改成 `Cp2x64x128b` 的试验触发了设备地址越界
并被立即隔离；另一个 `.0213` 变体在编译器中触发 `bad_variant_access`。最终源码
恢复 CUTLASS 4.7.0 官方 block-scaled 示例使用的
`Cp4x32x128bOp(CtaGroup.TWO)`，之后从 smoke 到两轮 1000 对正式运行均逐元素
正确。不要在当前 TMEM layout 上重试这两个 `Cp2` 变体。

## 正确性与执行证明

最终五轮正式运行均为 `PASS`：

- 所有输出有限；
- v13 与同轮 cuBLASLt 的 BF16 输出逐元素一致：`relative_l2=0.0`、
  `max_abs=0.0`；
- v13 与 cuBLASLt 相对原始 BF16 输入的 FP32 `torch.mm` reference 都是
  `relative_l2=0.03770923614501953`；
- 输出 dtype 为 BF16；
- 运行时报告 compute capability 10.3、74 个 active 2-CTA clusters、4096 个
  logical output clusters，以及每个 resident cluster 55 或 56 个 work tiles。

这端到端验证了所有 output tile 都得到正确结果；数值结果本身不能单独证明没有
发生内容相同的冗余写入，work 唯一性仍由 CUTLASS CLC scheduler contract 保证。
最终交付前本地 `py_compile` 和 v13/paired benchmark 的 9 项定向测试也全部通过。

## 复现命令

```bash
.venv/bin/modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v13 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 200 --iterations 1000
```

同口径对照：

```bash
.venv/bin/modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v11 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 200 --iterations 200

.venv/bin/modal run modal_dense_gemm.py \
  --implementation manual_pipeline_v12 \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 200 --iterations 200
```

## 原始 artifacts 与 Modal runs

- v13 200-pair round 1：
  [`20260909T053825.458984Z`](../artifacts/20260909T053825.458984Z-dense_gemm_mxfp8_16384_manual_pipeline_v13/result.json)
  (`ap-8kX4b4kpy5Bgus0ECsxtmn`)
- v13 200-pair round 2：
  [`20260909T053933.731709Z`](../artifacts/20260909T053933.731709Z-dense_gemm_mxfp8_16384_manual_pipeline_v13/result.json)
  (`ap-1qXK0tPBF7mGQ6wS3s9fFl`)
- v13 200-pair round 3：
  [`20260909T054047.426726Z`](../artifacts/20260909T054047.426726Z-dense_gemm_mxfp8_16384_manual_pipeline_v13/result.json)
  (`ap-sAnuDy0MKpfxrgkTQOKUZk`)
- v13 1000-pair round 1：
  [`20260909T054149.852027Z`](../artifacts/20260909T054149.852027Z-dense_gemm_mxfp8_16384_manual_pipeline_v13/result.json)
  (`ap-f8wXZo3f7FU0R6RIUU6hrY`)
- v13 1000-pair round 2，当前源码：
  [`20260909T055402.990915Z`](../artifacts/20260909T055402.990915Z-dense_gemm_mxfp8_16384_manual_pipeline_v13/result.json)
  (`ap-sLgTjvuGW4TluwpStwkHgd`)
- v12 paired control：
  [`20260909T054307.004287Z`](../artifacts/20260909T054307.004287Z-dense_gemm_mxfp8_16384_manual_pipeline_v12/result.json)
  (`ap-Uzev6m8UBft40wpk5eiDGx`)
- v11 paired control：
  [`20260909T054402.987366Z`](../artifacts/20260909T054402.987366Z-dense_gemm_mxfp8_16384_manual_pipeline_v11/result.json)
  (`ap-gpVZBMH1xsru4Am44cTqNb`)
- 最终配置的 SASS/PTX/CUBIN 快照：
  [`20260909T052220.425492Z`](../artifacts/20260909T052220.425492Z-dense_gemm_mxfp8_16384_manual_pipeline_v13/result.json)

## 设计参考

- NVIDIA CUTLASS 4.7.0 generic block-scaled persistent example：
  <https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py>
- NVIDIA CUTLASS CLC dynamic persistent example：
  <https://raw.githubusercontent.com/NVIDIA/cutlass/main/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py>
- NVIDIA CUTLASS pipeline implementation：
  <https://raw.githubusercontent.com/NVIDIA/cutlass/main/python/CuTeDSL/cutlass/pipeline/sm100.py>
- NVIDIA Transformer Engine source：
  <https://github.com/NVIDIA/TransformerEngine>
