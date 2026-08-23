# cutex

`cutex` 是一套面向 CuTeDSL 算子实验的最小脚手架：代码留在本机，编译、正确性验证、autotune 和性能测量发生在 Modal GPU 上，结构参考了 [`deciding/cutez`](https://github.com/deciding/cutez)。

当前自带 FP32 `C = A + B` 与 B300/SM103 rowwise MXFP8 dense GEMM 两个 CuTeDSL kernel，用它们验证整条链路：

- Modal 动态 GPU 选择：`L4`、`H100!`、`B200` 等；
- `cutex.autotune`：声明搜索空间，以 CUDA Events 实测候选项；
- `cutex.compile`：首次调优、按 shape 命中缓存、损坏缓存自动重建；
- `cutex.trace`：调用 CUTLASS IKET，采集 kernel 内部的 Perfetto 时间线；
- Modal Volume：按 GPU 型号长期保存调优缓存、结果、trace 和可选 IR；
- 本地 `artifacts/<run-id>-<kernel>/`：保存每次远端实验的结果副本，目录名同时包含时间和 kernel 名。

## 目录

```text
cutex/
├── cutex/
│   ├── autotune.py          # Config、decorator 与调优声明
│   ├── benchmark.py         # CUDA event 统计
│   ├── compiler.py          # 编译、搜索与 JSON cache
│   ├── trace.py             # CUTLASS IKET 编排与 trace 产物回传
│   ├── iket_worker.py       # IKET 独立单次 launch 工作负载
│   └── kernels/
│       ├── vector_add.py    # 基础 A+B CuTeDSL kernel
│       ├── dense_gemm.py    # 固定 16384³ 的 B300 MXFP8 Fprop
│       ├── dense_gemm_v0.py # 同一精度合同下的空白实现骨架
│       ├── dense_gemm_v1.py # 手工反量化、CUDA Core 分块 GEMM
│       ├── dense_gemm_v2.py # 手写 mbarrier 的单级 TMA + CUDA Core GEMM
│       └── dense_gemm_contract.py # shape、精度与 scale 布局合同
├── modal_app.py             # Modal image、GPU function、本地入口
├── modal_dense_gemm.py      # 固定 B300 MXFP8 GEMM 验证与 benchmark 入口
├── modal_torch_fp8_gemm.py  # B300 PyTorch tensor-wise FP8 GEMM benchmark
├── modal_te_mxfp8_gemm.py   # B300 Transformer Engine MXFP8 训练 GEMM benchmark
├── modal_te_mxfp8_4096.py   # B300 rowwise-only MXFP8 Fprop 性能上限
├── modal_te_mxfp8_protocol.py # 250 ms + 2 warmup + 单发 CUDA-event 协议
├── modal_te_mxfp8_sweep.py  # B300 TE MXFP8 的 1024³–32768³ shape sweep
├── modal_te_mxfp8_power_trace.py # B300 TE MXFP8 功率、时钟与输入模式追踪
├── modal_cublas_bf16_gemm.py # B300 直接 cuBLAS BF16/FP32-acc GEMM
├── cublas_bf16_gemm.cu      # cublasGemmEx CUDA benchmark
├── benchmarks/              # 可复现的基线报告
├── scripts/                 # A+B 与 dense GEMM 的 Windows 快捷入口
└── tests/                   # 无 GPU 也能运行的基础设施测试
```

## 1. 本地安装

需要 Python 3.11+ 和 [`uv`](https://docs.astral.sh/uv/)。CuTeDSL 本身只安装在 Modal 的 Linux image 中，因此 Windows 本机不需要 CUDA 或 PyTorch。

```powershell
uv sync --extra dev
uv run pytest
```

依赖已经固定为当前验证组合：Modal `1.5.4`（包含 `api-proxy-support`，可通过本机 HTTP/SOCKS 代理连接）、CuTeDSL `4.7.0`、PyTorch `2.13.0`。远端基础镜像是 `nvidia/cuda:13.0.2-devel-ubuntu22.04`。

## 2. 配置 Modal 凭据

Modal API 认证需要一对凭据：token ID 通常以 `ak-` 开头，token secret 通常以 `as-` 开头。只拿到其中一段无法认证。

推荐让 CLI 安全地读取并保存在用户目录的 `.modal.toml`，不要把 token 写进本仓库：

```powershell
uv run modal token set
uv run modal token info
```

也可以使用 `MODAL_TOKEN_ID` 与 `MODAL_TOKEN_SECRET` 环境变量。仓库的 `.gitignore` 已显式排除 `.env`、`.modal.toml` 和实验大文件。

## 3. 跑通 A+B

低成本 smoke test：

```powershell
uv run modal run modal_app.py --gpu L4 --m 2048 --n 2048 --warmup 5 --iterations 30
```

正式测 H100，`!` 可阻止 Modal 自动升级为 H200，适合可重复 benchmark：

```powershell
uv run modal run modal_app.py --gpu "H100!" --m 4096 --n 4096 --warmup 10 --iterations 100
```

测 B200：

```powershell
uv run modal run modal_app.py --gpu B200 --m 8192 --n 8192 --warmup 20 --iterations 200
```

Windows 也可以用封装脚本：

```powershell
.\scripts\run-modal.ps1 -Gpu L4 -M 2048 -N 2048 -Warmup 5 -Iterations 30
```

脚本会自动启用 Python UTF-8 输出，避免中文 Windows 的 GBK 控制台无法显示 Modal 状态符号。

第一次运行需要构建 image，并对 `32/64/128-bit` 三个 copy 配置逐一编译和计时；相同 GPU 与 shape 的后续运行直接从 Modal Volume 读取最优配置。强制重调：

```powershell
.\scripts\run-modal.ps1 -Gpu B200 -M 8192 -N 8192 -ForceRetune
```

## 4. 结果与 trace

成功时终端会输出类似：

```json
{
  "status": "PASS",
  "gpu": "NVIDIA L4",
  "shape": [2048, 2048],
  "best_config": {"name": "vector-128b", "kwargs": {"copy_bits": 128}},
  "median_us": 0.0,
  "effective_bandwidth_gbps": 0.0,
  "local_artifacts": ".../artifacts/<run-id>-vector_add"
}
```

这里的数值仅为格式示意，真实结果由远端 GPU 填写。每次实验包含：

- `result.json`：GPU/软件版本、正确性误差、候选配置、均值/中位数/P95、有效带宽；
- 可选 `iket/*.pftrace`：可拖入 [Perfetto UI](https://ui.perfetto.dev/)；
- 可选 `iket/*.trace.json` 与 `iket/run-iket.log`：IKET 原始 JSON 和采集日志；
- 远端同名副本：Modal Volume `cutex-autotune-cache`；
- 可选 CuTeDSL dump：加 `--dump-ir` 或脚本参数 `-DumpIr`。

性能与 trace 是两条分离的路径：autotune 和最终 benchmark 始终使用未启用 IKET 的编译结果，并用 CUDA Events 计时；加 `--trace` 后，benchmark 完成才会由独立子进程在 `run-iket` 下编译并单次 launch。这样 IKET 负责观察 kernel 内部阶段，CUDA Events 负责给出不受插桩影响的性能数字。`result.json` 会明确记录 `benchmark.instrumentation = "disabled"` 与 `trace.benchmark_instrumented = false`。

IKET 需要 SM90+，所以 L4 可做 benchmark，但不能加 `--trace`；H100/H200/B200/B300 可以。A+B trace 子任务默认把每个维度限制到最多 512。MXFP8 dense GEMM 的 kernel contract 固定为 16384³，因此它的 trace 也会采集完整 16384³ 单次 launch，产物可能很大；只看 TFLOPS 时不要启用 trace。示例：

```powershell
uv run modal run modal_app.py --gpu B200 --m 4096 --n 4096 --trace
uv run modal run modal_dense_gemm.py --m 16384 --n 16384 --k 16384 --trace

.\scripts\run-modal.ps1 -Gpu B200 -M 4096 -N 4096 -Trace
.\scripts\run-dense-gemm.ps1 -M 16384 -N 16384 -K 16384 -Trace
```

已在 B200 上验证 A+B trace 包含 `vector_add/setup/load/add/store`。MXFP8 dense GEMM 的 IKET ranges 为 `dense_gemm/prologue/mainloop/k_tile/epilogue`。

### 已验证基线（2026-08-21）

脚手架已在 Modal 的 NVIDIA L4（compute capability 8.9）上完成端到端验证：

| GPU | shape | autotune 结果 | median | P95 | 有效带宽 | 正确性 |
|---|---:|---:|---:|---:|---:|---:|
| L4 | 2048 × 2048 | `vector-128b`（持久化 cache 命中已复验） | 47.12 µs | — | 1068.16 GB/s | PASS |
| L4 | 4096 × 4096 | `vector-64b` | 875.01 µs | 887.81 µs | 230.09 GB/s | PASS，max abs error = 0 |
| B200 | 4096 × 4096 | `vector-64b` | 37.82 µs | 38.30 µs | 5322.72 GB/s | PASS，max abs error = 0 |
| B200 | 4096 × 4096 | `vector-64b`（持久化 cache 命中复验） | 37.25 µs | 37.41 µs | 5405.03 GB/s | PASS，max abs error = 0 |

2048² 的三份张量工作集接近 L4 的 L2 容量，因此其“有效带宽”主要反映 cache 命中，不能解释为 DRAM 物理带宽；4096² 的工作集更适合作为显存带宽参考。B200 两轮中位数平均为 37.54 µs，相对同尺寸 L4 约快 23.3 倍。验证环境为 CuTeDSL `4.7.0`、PyTorch `2.13.0+cu130`。

## 5. 跑固定 MXFP8 dense GEMM（B300）

实现位于 `cutex/kernels/dense_gemm.py`，数据布局及语义为：

```text
X:   row-major MXFP8 E4M3FN [16384, 16384]
W:   row-major MXFP8 E4M3FN [16384, 16384]
SFX: E8M0FNU，沿 K 每 32 个值一个，tcgen05 Swizzle32x4x4
SFW: E8M0FNU，沿 K 每 32 个值一个，tcgen05 Swizzle32x4x4
Y:   BF16 [16384, 16384] = X @ W.T
MMA: FP8 E4M3FN × E4M3FN，单一 FP32 TMEM accumulator
```

正式 kernel 只接受 `M=N=K=16384`，固定使用 `128 × 256 × 128` CTA tile、四级 A/B/scale TMA pipeline、SM103 `tcgen05` block-scaled MMA 和 128 threads。量化与 scale swizzle 在计时区外；没有 fast accumulation 或 split-K。`dense_gemm_v0.py` 保留同一 pointer 接口的纯 TODO 骨架；`dense_gemm_v1.py` 实现 16 × 16 × 16 shared-memory 分块、手工 MXFP8 反量化和标量 FP32 CUDA Core 累加；`dense_gemm_v2.py` 则按 K tile 严格串行执行四路 TMA、手写 mbarrier 等待、标量 FP32 CUDA Core 累加与 BF16 写回，不创建 `PipelineTmaAsync`、MMA 或 Tensor Core 对象。v1/v2 都用于教学与正确性对照，不以性能为目标。

```powershell
uv run modal run modal_dense_gemm.py --m 16384 --n 16384 --k 16384 `
  --gpu-warmup-seconds 10 --warmup 200 --iterations 1000
```

或使用 Windows 快捷入口：

```powershell
.\scripts\run-dense-gemm.ps1 -M 16384 -N 16384 -K 16384 `
  -GpuWarmupSeconds 10 -Warmup 200 -Iterations 1000
```

远端使用 Transformer Engine 只物化 rowwise MXFP8 数据和 GEMM-swizzled scales，再编译 CuTeDSL kernel。正确性同时对齐原始 BF16 输入的 FP32 `torch.mm` 与 TE Fprop；CUDA Events 分别测 custom kernel 和 TE，量化不进入计时。`result.json` 与可选 IKET 产物保存到本地 `artifacts/<run-id>-dense_gemm_mxfp8_16384/` 和 Modal Volume。加 `--trace`（脚本使用 `-Trace`）采集 kernel 内部阶段；加 `--dump-ir`（脚本使用 `-DumpIr`）可保留 IR/PTX。

### 已验证 dense GEMM 基线（2026-08-22）

2026-08-22 已在 `NVIDIA B300 SXM6 AC`（compute capability 10.3）、CuTeDSL `4.7.0`、Transformer Engine `2.17` 上验证 16384³。custom 输出与相同预量化输入上的 TE Fprop **逐元素一致**（relative L2 = 0），相对原始 BF16/FP32 参考的 relative L2 为 3.771%。10 秒 GPU warmup、200 次 kernel warmup、1000 个 CUDA-event samples 的持续性能结果为：

| 实现 | median | TFLOPS | 相对同轮 TE | 相对 4500 TFLOPS 硬件上限 |
|---|---:|---:|---:|---:|
| `cutex.kernels.dense_gemm` | **4473.264 µs** | **1966.37** | **74.57%** | **43.70%** |
| 同轮 Transformer Engine | 3335.792 µs | 2636.88 | 100.00% | 58.60% |

当前最简实现比同 shape、同轮 TE 低约 25.43% 吞吐；它已完全对齐 shape、存储与精度合同，但还没有 persistent scheduling、cluster/2-CTA、TMA store 等峰值优化。单次冷态样本受瞬时 boost 影响，不作为正式回归值。

CUDA Core 教学版本可通过 `--implementation cuda_core_v1` 选择。它以 `16 × 16 × 16` 分块手工反量化 MXFP8，并用标量 FP32 乘加计算；B300 上完整 16384³ 正确性已通过，单发约 2346.9 ms / 3.748 TFLOPS。设计与原始结果见 [`benchmarks/B300_DENSE_GEMM_V1_CUDA_CORE.md`](benchmarks/B300_DENSE_GEMM_V1_CUDA_CORE.md)。

手写 TMA 同步版本可通过 `--implementation tma_cuda_core_v2` 选择。它使用 `16 × 16 × 128` A/B tile、原生 `128 × 128` scale TMA tile，以及单个在 phase 0/1 间轮换的 mbarrier；每轮执行 `arrive_and_expect_tx → 4×TMA → wait → CUDA Core FMA → CTA sync`。2026-08-24 已在 B300 上通过完整 16384³ 正确性验证；无 warmup 的单个 CUDA-event 样本为 9614.4 ms / 0.915 TFLOPS，仅用于 smoke test，不作为性能基线。

## 6. B300 MXFP8 rowwise Fprop 性能上限

仓库只保留一组用于后续 `dense_gemm` 对齐的正式性能基线：4096³ Fprop，逻辑运算为 `Y = X @ W.T`。两个输入都是 rowwise-only MXFP8 E4M3，每 32 个值使用一个 E8M0 scale；乘法为 FP8，关闭 fast accumulation、使用 FP32 compute/accumulate contract，输出 BF16。量化和 scale swizzle 在计时区外完成。

```powershell
uv run modal run modal_te_mxfp8_4096.py::main `
  --size 4096 --warmup 200 --iterations 1000 `
  --repeats 10 --gpu-warmup-seconds 10
```

2026-08-22 在 `NVIDIA B300 SXM6 AC`、CUDA 13.3、Transformer Engine 2.17 上的结果。由 NVIDIA 官方整机规格换算的单卡 dense FP8 Tensor Core roofline 为 4500 TFLOPS，对应 4096³ GEMM 的理论最短延迟 30.542 µs：

| shape (M×N×K) | 中位延迟 | 吞吐 | 硬件峰值达成率 | relative L2 vs FP32 | 结果 |
|---:|---:|---:|---:|---:|---:|
| 4096×4096×4096 | **56.288 µs** | **2441.70 TFLOPS** | **54.26%** | 3.771% | PASS |

后续 kernel 的正式目标是达到 NVIDIA 实测基线，即不高于 56.288 µs、不低于 2441.70 TFLOPS；4500 TFLOPS 是硬件绝对上限。这是预量化 raw GEMM 口径，不包含量化、attention、归一化、激活、优化器或通信。完整精度合同、理论上限推导、验收公式、重复组数据、环境和原始 artifact 见 [`benchmarks/B300_MXFP8_ROWWISE_FPROP_4096.md`](benchmarks/B300_MXFP8_ROWWISE_FPROP_4096.md)。

1024³ 到 32768³ 的同精度 Transformer Engine shape sweep 及复现命令见 [`benchmarks/B300_MXFP8_TE_SHAPE_SWEEP_1024_32768.md`](benchmarks/B300_MXFP8_TE_SHAPE_SWEEP_1024_32768.md)。这组 sweep 用于观察尺寸扩展趋势，不替代上面的正式 4096³ 重复组基线。

2026-08-23 按同一最初口径重新执行的 cuBLASLt-backed MXFP8 sweep 见 [`benchmarks/B300_CUBLASLT_MXFP8_SHAPE_SWEEP_RERUN_20260823.md`](benchmarks/B300_CUBLASLT_MXFP8_SHAPE_SWEEP_RERUN_20260823.md)。4096³ 为 2886.40 TFLOPS；8192³–32768³ 为 2629.71–2681.42 TFLOPS。该路径由 TE 准备 MXFP8 数据并调用 `cublasLtMatmul`，不是自定义 CuTeDSL kernel。

关闭 split accumulator 的 fast-accum 请求复测见 [`benchmarks/B300_MXFP8_TE_FAST_ACCUM_SWEEP_1024_32768.md`](benchmarks/B300_MXFP8_TE_FAST_ACCUM_SWEEP_1024_32768.md)。该模式仍由 TE 固定为 `CUBLAS_COMPUTE_32F`，不能标注为 FP16 accumulator。

16384³ 的功率/时钟追踪见 [`benchmarks/B300_MXFP8_TE_POWER_TRACE_16384.md`](benchmarks/B300_MXFP8_TE_POWER_TRACE_16384.md)。相同 TE kernel 在 `zero`/`one` 输入上达到约 4479 TFLOPS 和 2032 MHz；正态随机输入触发 1100 W software power cap，持续性能降至约 2657 TFLOPS、SM 时钟约 1125 MHz。

直接 `cublasGemmEx` 的 16384³ BF16 输入输出、FP32 累加测试见 [`benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_16384.md`](benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_16384.md)。random uniform 输入持续约 1490 TFLOPS；all-one 控制组约 2239 TFLOPS，达到单卡 2250 TFLOPS dense BF16 roofline 的 99.5%。

同一精度合同的 4096³ 结果见 [`benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_4096.md`](benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_4096.md)：random uniform 输入约 1369 TFLOPS，all-one 约 1888 TFLOPS。

4096³ random 输入的 10 次单发、每次间隔 1 秒测试见 [`benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_4096_SPACED.md`](benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_4096_SPACED.md)。该模式未触发 power cap，但每次都是空闲后的冷态 launch，中位性能为 864.79 TFLOPS。

进一步对 warmup 数和样本间隔做 96 点协议搜索的结果见 [`benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_4096_PROTOCOL_TUNING.md`](benchmarks/B300_CUBLAS_BF16_FP32ACC_GEMM_4096_PROTOCOL_TUNING.md)。推荐口径为每个样本前等待 250 ms、执行 2 发不计时 random warmup，再测 1 发；10 次中位为 1614.95 TFLOPS，5 ms telemetry 未观察到 power cap。

同一推荐口径下的 4096³ rowwise-only MXFP8 Fprop 实测见 [`benchmarks/B300_MXFP8_TE_4096_RECOMMENDED_PROTOCOL.md`](benchmarks/B300_MXFP8_TE_4096_RECOMMENDED_PROTOCOL.md)：10 次单发中位为 **49.216 µs / 2792.57 TFLOPS**，达到 4500 TFLOPS roofline 的 **62.06%**；335 个 5 ms telemetry samples 未观察到持续 power cap。该结果是短突发口径，原先 2441.70 TFLOPS 的连续重复组仍作为持续性能基线。

## 7. 给新 kernel 接入 autotune

```python
import cutex
import cutlass
import cutlass.cute as cute

CONFIGS = [
    cutex.Config({"tile": 64}, name="tile64"),
    cutex.Config({"tile": 128}, name="tile128"),
]

@cutex.autotune(configs=CONFIGS, key=["m", "n"], warmup=5, rep=30)
@cute.jit
def my_op(a, out, m: cutlass.Int32, n: cutlass.Int32,
          tile: cutlass.Constexpr = 128):
    # launch your @cute.kernel here
    ...

compiled = cutex.compile(
    my_op,
    a,
    out,
    m,
    n,
    cache_path="/cutex-cache/autotune/my_gpu/my_op.json",
)
compiled(a, out, m, n)
```

缓存条目同时带 kernel 标识、`key` 值和完整配置；搜索空间变化后，旧配置不再匹配并会自动重调。

## 设计来源

- [`deciding/cutez`](https://github.com/deciding/cutez)：`Config`/decorator/compile cache 与 GPU trace 的整体分层；
- [`deciding/cutez/dense_gemm_1.py`](https://github.com/deciding/cutez/blob/main/modal/blackwell/dense_gemm_1.py)：最简 B200 dense GEMM 的 TMA、低级 mbarrier、UMMA/TMEM 与 epilogue 数据路径；
- [NVIDIA CUTLASS IKET profiling guide](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/guides/iket_profiling.html)：`run-iket`、kernel 内 range 与 Perfetto/JSON 产物；
- [NVIDIA CUTLASS IKET GEMM example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/dsl_tutorials/fp16_gemm_4_iket.py)：CuTeDSL kernel 内 IKET 标记方式；
- [NVIDIA CUTLASS CuTeDSL elementwise add](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/ampere/kernel/elementwise/elementwise_add.py)：A+B 的 TV-layout、predication、gmem→rmem→gmem 模式；
- [NVIDIA Transformer Engine MXFP8](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)：Blackwell MXFP8 recipe、数据格式与 shape 约束；
- [NVIDIA Transformer Engine GEMM profiling](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/gemm_profiling/gemm_profiling.html)：训练 GEMM shape 推导、autocast 与 prequantized 测量口径；
- [Modal CUDA 文档](https://modal.com/docs/guide/cuda)：远端 CUDA image 与 GPU 驱动边界；
- [Modal GPU 文档](https://modal.com/docs/guide/gpu)：GPU 型号和动态选择。
