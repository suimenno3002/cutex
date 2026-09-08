# 在 Modal B300 上运行 cutex

本文记录从本地仓库出发，经 Modal 构建远端镜像、申请单张 B300、编译并验证 CuTeDSL kernel、回收结果和停止异常任务的完整流程。

适用快照：2026-09-07。本文以仓库当前的 `modal_dense_gemm.py` 和 Modal `1.5.4` 为准。

## 验证边界

- 当前可运行路径包括默认 `tensor_core` 和手写调度的 `manual_pipeline_v9/v10/v11/v12`；默认实现仍是稳定主路径，后四者用于 2SM pipeline 对照和性能实验。
- v9 保留四级 ring 与行主序 grid；v10 是六级 ring 加 Morton swizzle；v11 冻结六级流水，只改用 CuTe layout 的 `8×8` cluster block swizzle；v12 保持 v11 的 tile/ring/swizzle，按 CUDA cluster occupancy 发射静态 persistent grid（此前 B300 记录值为 74 个 cluster）并重叠 MMA 与 epilogue。
- `tma_cuda_core_v2` 曾完成过一次 16384³ smoke test；2026-08-24 最近一次含 layout 日志的复测在首个 kernel launch 后长期占满 GPU，未返回正确性或计时结果。当前应把它视为待排查回归，而不是稳定入口。
- 本地测试或 JIT 编译日志不等于 B300 成功。只有远端结果同时给出 B300/SM103、`status: PASS`、正确性数据和非零计时，才算端到端跑通。

## 1. 运行链路

```text
本地仓库
  └─ uv run modal run modal_dense_gemm.py ...
       ├─ Modal 构建/复用 nvcr.io/nvidia/pytorch:26.07-py3
       ├─ 将本地 cutex Python package 复制进镜像
       ├─ 申请 gpu="B300" 的临时容器
       ├─ 在 B300 上量化、JIT 编译、运行、验正确性和计时
       ├─ 将 result.json 写入 Modal Volume
       └─ 将结果返回本地 artifacts/<run-id>-<kernel>/result.json
```

不需要在远端手工克隆仓库，也不需要先 `modal deploy`。`modal run` 创建临时 App，正常完成或本地调用进程退出后会停止。当前入口的关键环境为：

| 项目 | 当前值 |
|---|---|
| Modal GPU 请求 | 单张 `B300` |
| 远端镜像 | `nvcr.io/nvidia/pytorch:26.07-py3` |
| 镜像内 CUDA | 13.3.1 |
| Transformer Engine | 镜像自带 2.17 |
| CuTeDSL | `nvidia-cutlass-dsl[cu13]==4.7.0` |
| 目标架构 | B300/SM103，运行时强制校验 compute capability 10.3 |
| 单次函数超时 | 30 分钟 |
| 持久化 Volume | `cutex-autotune-cache`，挂载到 `/cutex-cache` |

Modal 当前要求 B300 镜像使用 CUDA 13.1 或更高版本；NVIDIA PyTorch 26.07 镜像的 CUDA 13.3.1 满足这个条件。不要把 B300 入口换回 `modal_app.py` 使用的 CUDA 13.0.2 通用镜像。

## 2. 本地准备

需要 Python 3.11+、`uv` 和能够计费的 Modal 账号。本机不需要 NVIDIA GPU、CUDA、PyTorch 或 CuTeDSL；这些依赖在远端镜像里。

在仓库根目录执行：

```bash
uv sync --locked --extra dev
uv run pytest
uv run modal run modal_dense_gemm.py --help
```

运行使用的是当前工作区内容，包括未提交修改。为保证结果可追溯，每次申请 GPU 前至少记录：

```bash
git rev-parse HEAD
git status --short
```

当前 `result.json` 不会自动写入 Git commit 或 diff；工作区不干净时，应把上述输出与结果一同保存。

## 3. 配置 Modal 凭据和预算

第一次使用可按 Modal 官方流程登录：

```bash
uv run modal setup
uv run modal token info
```

如果已有 token ID 和 token secret，也可使用交互式录入：

```bash
uv run modal token set
uv run modal token info
```

不要把 token 写入仓库、命令行历史、Markdown 或实验产物。仓库已经忽略 `.modal.toml`、`.env` 和 `.env.*`。

申请 B300 前查询当前费率，并在 Modal Workspace 中设置合适的 spend budget：

```bash
uv run modal billing rates
```

Modal 按实际资源使用时间计费，价格会变化，因此本文不固化单价。第一次运行还会经历镜像构建；正式测试前先做下面的低成本预检和单样本 smoke test。

## 4. 低成本链路预检

先用 L4 跑仓库的 A+B kernel：

```bash
uv run modal run modal_app.py --gpu L4 --m 2048 --n 2048 --warmup 5 --iterations 30
```

这一步验证：

- Modal 凭据、网络和 Workspace 可用；
- 镜像能构建，本地 `cutex` 源码能上传；
- CuTeDSL 能在远端 JIT；
- kernel 正确性和 CUDA Events 计时通过；
- 远端 Volume 和本地 artifact 都能写入。

终端必须出现 `status: PASS`、`gpu: NVIDIA L4`、非零 `median_us` 和本地 artifact 路径；打开对应 `result.json` 后还应看到正确性数据。它只证明通用链路，不证明 B300/SM103 kernel 正确。

不要把这条命令的 `--gpu` 改为 B300：`modal_app.py` 当前使用 CUDA 13.0.2，而 Modal 的 B300 要求 CUDA 13.1+。B300 必须走下一节的 `modal_dense_gemm.py`。

## 5. B300 最小端到端 smoke test

正式入口固定申请 B300，不提供 `--gpu` 参数。第一次先使用默认 Tensor Core 实现、零预热和一个计时样本：

```bash
uv run modal run modal_dense_gemm.py \
  --implementation tensor_core \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 0 --warmup 0 --iterations 1
```

该 kernel 的合同固定为 `M=N=K=16384`，不能通过减小 shape 做 smoke test。即使只设一个计时样本，入口仍会完成：

1. 检查 CUDA、MXFP8 和实际 GPU compute capability；
2. 生成 BF16 输入并用 Transformer Engine 量化为 rowwise MXFP8；
3. 在 B300 上 JIT 编译当前 CuTeDSL kernel；
4. 各运行一次 custom kernel 和 TE 对照；
5. 用原始 BF16 输入的 FP32 `torch.mm` 做正确性参考；
6. 分别测 custom kernel 和 TE 的 CUDA-event 时间；
7. 写入远端 Volume，返回并保存本地 `result.json`。

第一次构建 NVIDIA 镜像和安装 CuTeDSL 可能明显慢于后续运行。镜像构建完成并不代表 kernel 已通过，必须等待最终 JSON。

### 成功门槛

最终输出和 `result.json` 至少应满足：

- `status == "PASS"`；
- `environment.gpu_name` 是实际 NVIDIA B300；
- `environment.compute_capability == "10.3"`；
- `environment.mxfp8_available == true`；
- `correctness.all_finite == true`；
- `correctness.custom_relative_l2_vs_fp32 < 0.15`；
- `benchmark.kernel.median_us > 0` 且 `benchmark.kernel.tflops > 0`；
- 终端打印出存在的 `local_artifact` 路径。

只看到 App URL、容器启动、GPU 100% 或 CuTe layout/JIT 日志，都不能代替这些门槛。

## 6. 正式性能测试

smoke test 通过后，再使用仓库历史基线相同的持续测量口径：

```bash
uv run modal run modal_dense_gemm.py \
  --implementation tensor_core \
  --m 16384 --n 16384 --k 16384 \
  --gpu-warmup-seconds 10 --warmup 200 --iterations 1000
```

默认 `tensor_core` 路径会把输入量化排除在计时区外，分别报告 custom kernel 和同轮 Transformer Engine Fprop。仓库 2026-08-22 的历史 B300 基线约为：

| 实现 | median | 吞吐 |
|---|---:|---:|
| custom `tensor_core` | 4473.264 µs | 1966.37 TFLOPS |
| 同轮 Transformer Engine | 3335.792 µs | 2636.88 TFLOPS |

这些值只用于发现数量级错误或明显回退，不是新实例必须逐位复现的硬阈值。输入分布、功耗限制、时钟、驱动、镜像和云端实例状态都会影响 B300 性能。

Windows PowerShell 也可使用仓库脚本：

```powershell
.\scripts\run-dense-gemm.ps1 -M 16384 -N 16384 -K 16384 `
  -GpuWarmupSeconds 10 -Warmup 200 -Iterations 1000
```

该脚本没有暴露 `--implementation`，因此使用默认 `tensor_core`，正好对应推荐主路径。

## 7. 当前实现选择

| `--implementation` | 用途 | 当前状态 |
|---|---|---|
| `tensor_core` | SM103 block-scaled Tensor Core 主实现 | 推荐；B300 历史正确性和性能基线已存在 |
| `cuda_core_v1` | 手工反量化和标量 CUDA Core 教学对照 | B300 历史正确性已通过；约 2.35 秒/发，不用于性能目标 |
| `tma_cuda_core_v2` | 手写 TMA + mbarrier 教学对照 | 有历史单次 PASS，但最近回归挂起；先排查再重跑 |
| `manual_pipeline_v9` | 原始 2SM、手写 mbarrier、warp-specialized Tensor Core 实现 | 四级 ring + 行主序 grid，非 persistent |
| `manual_pipeline_v10` | 从修改后 v9 独立出的后继实现 | 六级 ring + Z-order cluster tile，非 persistent |
| `manual_pipeline_v11` | CuTe layout thread-block swizzle 对照 | 六级 ring + `8×8` cluster block swizzle，非 persistent |
| `manual_pipeline_v12` | v11 的 persistent 后继版本 | occupancy-sized 常驻 2-CTA grid + 原生静态调度 + overlapping accumulator；等待 B300 实测闭环 |

如需复核 v1，只使用最小口径：

```bash
uv run modal run modal_dense_gemm.py \
  --implementation cuda_core_v1 \
  --gpu-warmup-seconds 0 --warmup 0 --iterations 1
```

不要把 v1/v2 的慢速结果与 Tensor Core 性能基线混在一起。

## 8. 结果和产物

成功返回后，本地结果位于：

```text
artifacts/<UTC-run-id>-<kernel-name>/result.json
```

远端副本位于 Modal Volume：

```text
cutex-autotune-cache
└── runs/NVIDIA-B300-SXM6-AC/<UTC-run-id>/<kernel-name>/result.json
```

列出或取回远端结果：

```bash
uv run modal volume ls cutex-autotune-cache runs/NVIDIA-B300-SXM6-AC
uv run modal volume get cutex-autotune-cache \
  runs/NVIDIA-B300-SXM6-AC/<run-id>/<kernel-name>/result.json \
  artifacts/recovered-result.json
```

`result.json` 记录合同、实现配置、正确性、CUDA-event 统计、编译耗时、GPU/软件版本和远端目录。若远端函数在写结果前异常或挂起，本地 artifact 不会生成。

### IR 和 trace

- `--dump-ir` 将 CuTeDSL dump/PTX 保存在该次远端 run 目录的 `cute-dsl-dump/`，需要通过 Volume 取回。
- `--trace` 会在普通 benchmark 完成后额外运行一次完整 16384³ IKET 采集，成本和产物都更大；第一次 smoke 和普通 TFLOPS 测试不要启用。
- `cutex/iket_worker.py` 会按 `--implementation` 选择 `tensor_core` 或 `manual_pipeline_v9/v10/v11/v12`。v9-v11 的 trace 由中部 cluster `(32,32,0)` 写 ranges；v12 改由常驻物理 cluster `(0,0,0)` 记录多个 work tile。四者都把每 warp event buffer 提高到 2048；v1/v2 仍不支持 `--trace`。

## 9. 观察和停止异常任务

`modal run` 会在终端打印 App URL/ID。另开终端可查看状态和日志：

```bash
uv run modal app list
uv run modal app logs <app-id> --timestamps --tail 200
```

普通前台运行按 `Ctrl+C` 后，临时 App 应自动停止；随后仍要用 `app list` 确认。若容器仍在运行，显式停止：

```bash
uv run modal app stop <app-id>
```

第一次验证不要加 `modal run --detach`。若 kernel 已进入 GPU 执行，长时间没有正确性、计时或阶段进展，应停止任务并先缩小诊断范围；不要仅依赖 30 分钟 timeout 来控制费用。

对 `tma_cuda_core_v2`，如果打印完 layout 后 GPU 长期 100% 且没有返回，按已知回归处理：保存 App ID、容器 ID 和带时间戳日志，停止任务，然后对首个 launch/mbarrier 阶段加诊断；不要直接反复提交完整 16384³。

## 10. 常见问题

| 现象 | 检查和处理 |
|---|---|
| `token_invalid`、未登录或 Workspace 错误 | 重新执行 `uv run modal setup`，再用 `uv run modal token info` 确认；不要提交凭据文件 |
| B300 长时间排队 | 查看 Modal App 状态和容量后重试；不要改成 `B200+`，当前 kernel 会强制拒绝非 SM103 GPU |
| 报 shape 不合法 | 恢复 `16384 16384 16384`；正式 kernel 是固定合同 |
| `MXFP8 is unavailable` | 确认仍使用 PyTorch 26.07 镜像及其 Transformer Engine，且实际 GPU 为 B300 |
| 只看到 image/JIT 成功 | 继续等最终 `status: PASS`、正确性和 CUDA-event 数据；编译成功不是执行成功 |
| 没有本地 artifact | 远端调用没有正常返回；查 App 日志和 Volume，不能把该轮记为 PASS |
| 性能异常偏高或偏低 | 先核对实现、shape、warmup、iterations、输入分布、GPU 型号和计时器，再与同轮 TE 对照 |
| `tma_cuda_core_v2` 卡住 | 立即按上一节保留日志并停止；该路径当前有未闭环的 mbarrier/首发挂起回归 |

## 11. 每次运行后的检查清单

- [ ] 记录 Git commit 和工作区状态。
- [ ] `modal token info` 指向预期 Workspace。
- [ ] 已查看当前 B300 费率和 Workspace budget。
- [ ] 首次环境先通过 L4 A+B 预检。
- [ ] B300 smoke 使用目标实现、16384³、0/0/1 口径；不确定时先用默认 `tensor_core`。
- [ ] 结果明确是 NVIDIA B300、SM103、`status: PASS`。
- [ ] 正确性和 CUDA-event 性能字段齐全。
- [ ] 本地 `result.json` 已保存，必要时从 Volume 备份额外产物。
- [ ] `modal app list` 中没有遗留的运行中临时 App。
- [ ] smoke 通过后才运行 10 秒 GPU warmup、200 warmup、1000 samples 的正式测试。

## 参考

- 仓库入口：[`modal_dense_gemm.py`](modal_dense_gemm.py)
- 低成本预检入口：[`modal_app.py`](modal_app.py)
- 固定 shape/精度合同：[`cutex/kernels/dense_gemm_contract.py`](cutex/kernels/dense_gemm_contract.py)
- v1 历史验证：[`benchmarks/B300_DENSE_GEMM_V1_CUDA_CORE.md`](benchmarks/B300_DENSE_GEMM_V1_CUDA_CORE.md)
- 现有主说明和 B300 基线：[`README.md`](README.md)
- [Modal 入门与认证](https://modal.com/docs/guide)
- [Modal GPU/B300 说明](https://modal.com/docs/guide/gpu)
- [Modal `run` CLI](https://modal.com/docs/cli/latest/run)
- [Modal Volume](https://modal.com/docs/guide/volumes)
- [Modal App 查看与停止](https://modal.com/docs/cli/latest/app)
- [Modal 计费](https://modal.com/docs/guide/billing)
- [NVIDIA PyTorch 26.07 release notes](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-26-07.html)
