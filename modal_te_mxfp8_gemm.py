"""Benchmark training GEMM shapes with Transformer Engine MXFP8 on Modal B300."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-te-mxfp8-gemm"
CACHE_MOUNT = "/cutex-cache"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
KERNEL_NAME = "transformer_engine_mxfp8_gemm"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("cutex-autotune-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.07-py3")
    .entrypoint([])
    .env(
        {
            "NVIDIA_IMEX_CHANNELS": "0",
            "PYTHONUNBUFFERED": "1",
        }
    )
)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


@app.function(image=image, gpu="B300", timeout=15 * 60)
def probe_remote() -> dict:
    import subprocess

    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling

    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")
    availability = te.is_mxfp8_available(return_reason=True)
    is_available, reason = (
        availability if isinstance(availability, tuple) else (bool(availability), "")
    )
    if not is_available:
        raise RuntimeError(f"Transformer Engine MXFP8 is unavailable: {reason}")

    torch.manual_seed(20260822)
    layer = te.Linear(
        4096,
        4096,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
    )
    x = torch.randn((256, 4096), device="cuda", dtype=torch.bfloat16)
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
    with te.autocast(enabled=True, recipe=recipe):
        y = layer(x)
    torch.cuda.synchronize()

    properties = torch.cuda.get_device_properties(0)
    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "status": "PASS",
        "gpu": str(properties.name),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "torch": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "transformer_engine": str(transformer_engine.__version__),
        "mxfp8_available": bool(is_available),
        "mxfp8_reason": str(reason),
        "output_shape": list(y.shape),
        "output_dtype": str(y.dtype),
        "output_all_finite": bool(torch.isfinite(y).all().item()),
        "nvidia_smi": smi,
    }


@app.function(
    image=image,
    gpu="B300",
    timeout=30 * 60,
    volumes={CACHE_MOUNT: cache_volume},
)
def run_remote(
    hidden_size: int = 4096,
    intermediate_size: int = 16384,
    num_attention_heads: int = 32,
    num_hidden_layers: int = 24,
    micro_batch_size: int = 31,
    sequence_length: int = 512,
    warmup: int = 10,
    iterations: int = 100,
    repeats: int = 3,
    gpu_warmup_seconds: float = 5.0,
    include_prequantized: bool = True,
) -> dict:
    import gc
    import statistics
    import subprocess
    import time

    import torch
    import torch.nn.functional as F
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.common.recipe import Format, MXFP8BlockScaling

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")
    if min(
        hidden_size,
        intermediate_size,
        num_attention_heads,
        num_hidden_layers,
        micro_batch_size,
        sequence_length,
    ) <= 0:
        raise ValueError("all model dimensions must be positive")
    if hidden_size % num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if warmup < 0 or iterations < 1 or repeats < 1 or gpu_warmup_seconds < 0:
        raise ValueError("invalid benchmark iteration or warmup setting")

    tokens = micro_batch_size * sequence_length
    dimensions = (tokens, hidden_size, intermediate_size, 3 * hidden_size)
    if any(dimension % 32 for dimension in dimensions):
        raise ValueError("MXFP8 requires all derived GEMM dimensions to be divisible by 32")

    availability = te.is_mxfp8_available(return_reason=True)
    is_available, reason = (
        availability if isinstance(availability, tuple) else (bool(availability), "")
    )
    if not is_available:
        raise RuntimeError(f"Transformer Engine MXFP8 is unavailable: {reason}")

    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    remote_run_dir = (
        Path(CACHE_MOUNT)
        / "runs"
        / _safe_slug(gpu_name)
        / run_id
        / KERNEL_NAME
    )
    remote_run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260822)
    device = torch.device("cuda")
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    # Confirm that both the forward and backward training paths execute.
    check_layer = te.Linear(
        1024,
        1024,
        bias=False,
        params_dtype=torch.bfloat16,
        device=device,
    )
    check_x = torch.randn(
        (128, 1024),
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference = F.linear(check_x.detach().float(), check_layer.weight.detach().float())
    with te.autocast(enabled=True, recipe=recipe):
        check_y = check_layer(check_x)
        check_loss = check_y.float().square().mean()
    difference = check_y.float() - reference
    relative_l2_error = float(
        (
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference)
        ).detach()
    )
    check_loss.backward()
    torch.cuda.synchronize()
    correctness = {
        "shape": [128, 1024, 1024],
        "forward_all_finite": bool(torch.isfinite(check_y).all().item()),
        "input_gradient_all_finite": bool(torch.isfinite(check_x.grad).all().item()),
        "weight_gradient_all_finite": bool(
            torch.isfinite(check_layer.weight.grad).all().item()
        ),
        "forward_relative_l2_error_vs_fp32": relative_l2_error,
    }
    if not all(
        correctness[key]
        for key in (
            "forward_all_finite",
            "input_gradient_all_finite",
            "weight_gradient_all_finite",
        )
    ) or relative_l2_error >= 0.15:
        raise AssertionError(f"MXFP8 training validation failed: {correctness}")
    del check_layer, check_x, check_y, check_loss, reference, difference
    gc.collect()
    torch.cuda.empty_cache()

    H, I, M = hidden_size, intermediate_size, tokens
    shape_groups = {
        "fprop": [
            ("QKV Proj", M, H, 3 * H),
            ("Attn Out", M, H, H),
            ("MLP Up", M, H, I),
            ("MLP Down", M, I, H),
        ],
        "dgrad": [
            ("QKV Proj", M, 3 * H, H),
            ("Attn Out", M, H, H),
            ("MLP Up", M, I, H),
            ("MLP Down", M, H, I),
        ],
        "wgrad": [
            ("QKV Proj", H, M, 3 * H),
            ("Attn Out", H, M, H),
            ("MLP Up", H, M, I),
            ("MLP Down", I, M, H),
        ],
    }

    def warm_gpu(duration_seconds: float) -> None:
        if duration_seconds <= 0:
            return
        a = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        b = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        torch.cuda.synchronize()
        until = time.perf_counter() + duration_seconds
        while time.perf_counter() < until:
            for _ in range(10):
                torch.matmul(a, b)
            torch.cuda.synchronize()
        del a, b
        torch.cuda.empty_cache()

    def time_cuda(run_fn, leading_fn) -> dict:
        samples_ms = []
        for _ in range(repeats):
            leading_fn()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                run_fn()
            end_event.record()
            end_event.synchronize()
            samples_ms.append(float(start_event.elapsed_time(end_event)) / iterations)
        latency_ms = statistics.median(samples_ms)
        return {
            "latency_ms": latency_ms,
            "mean_latency_ms": statistics.fmean(samples_ms),
            "min_latency_ms": min(samples_ms),
            "max_latency_ms": max(samples_ms),
            "repeat_average_latency_ms": samples_ms,
        }

    def with_throughput(stats: dict, flops: int) -> dict:
        return {
            **stats,
            "tflops": flops / stats["latency_ms"] / 1e9,
        }

    def benchmark_bf16(m: int, k: int, n: int) -> dict:
        a = torch.randn((m, k), device=device, dtype=torch.bfloat16)
        b = torch.randn((k, n), device=device, dtype=torch.bfloat16)
        lead_a = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        lead_b = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)

        def run_fn():
            torch.matmul(a, b)

        for _ in range(warmup):
            run_fn()
        torch.cuda.synchronize()
        stats = time_cuda(run_fn, lambda: torch.matmul(lead_a, lead_b))
        del a, b, lead_a, lead_b
        return stats

    def benchmark_mxfp8_autocast(m: int, k: int, n: int) -> dict:
        linear = te.Linear(
            k,
            n,
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
        )
        x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
        lead_linear = te.Linear(
            4096,
            4096,
            bias=False,
            params_dtype=torch.bfloat16,
            device=device,
        )
        lead_x = torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)

        with te.autocast(enabled=True, recipe=recipe):
            for _ in range(warmup):
                linear(x)
            torch.cuda.synchronize()
            stats = time_cuda(lambda: linear(x), lambda: lead_linear(lead_x))
        del linear, x, lead_linear, lead_x
        return stats

    def benchmark_mxfp8_prequantized(m: int, k: int, n: int) -> dict:
        quantizer = te.MXFP8Quantizer(tex.DType.kFloat8E4M3)
        a_q = quantizer.quantize(
            torch.randn((k, m), device=device, dtype=torch.bfloat16)
        )
        b_q = quantizer.quantize(
            torch.randn((k, n), device=device, dtype=torch.bfloat16)
        )
        output = torch.empty((n, m), device=device, dtype=torch.bfloat16)
        workspace_size = 32 * 1024 * 1024
        workspace = torch.empty(workspace_size, device=device, dtype=torch.uint8)
        lead_a_q = quantizer.quantize(
            torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        )
        lead_b_q = quantizer.quantize(
            torch.randn((4096, 4096), device=device, dtype=torch.bfloat16)
        )
        lead_output = torch.empty(
            (4096, 4096), device=device, dtype=torch.bfloat16
        )

        def gemm(a, b, out):
            tex.generic_gemm(
                a,
                False,
                b,
                True,
                out,
                None,
                tex.DType.kBFloat16,
                None,
                tex.DType.kBFloat16,
                False,
                None,
                False,
                workspace,
                workspace_size,
                False,
                False,
            )

        for _ in range(warmup):
            gemm(a_q, b_q, output)
        torch.cuda.synchronize()
        stats = time_cuda(
            lambda: gemm(a_q, b_q, output),
            lambda: gemm(lead_a_q, lead_b_q, lead_output),
        )
        del a_q, b_q, output, workspace, lead_a_q, lead_b_q, lead_output
        return stats

    def aggregate(entries: list[dict]) -> dict:
        passes = {}
        for pass_name in shape_groups:
            selected = [entry for entry in entries if entry["pass"] == pass_name]
            total_flops = sum(entry["flops"] for entry in selected)
            bf16_ms = sum(entry["bf16"]["latency_ms"] for entry in selected)
            mxfp8_ms = sum(entry["mxfp8"]["latency_ms"] for entry in selected)
            passes[pass_name] = {
                "flops": total_flops,
                "bf16_latency_ms": bf16_ms,
                "mxfp8_latency_ms": mxfp8_ms,
                "bf16_effective_tflops": total_flops / bf16_ms / 1e9,
                "mxfp8_effective_tflops": total_flops / mxfp8_ms / 1e9,
                "mxfp8_speedup_vs_bf16": bf16_ms / mxfp8_ms,
            }
        total_flops = sum(entry["flops"] for entry in entries)
        bf16_ms = sum(entry["bf16"]["latency_ms"] for entry in entries)
        mxfp8_ms = sum(entry["mxfp8"]["latency_ms"] for entry in entries)
        return {
            "passes": passes,
            "per_layer": {
                "flops": total_flops,
                "bf16_latency_ms": bf16_ms,
                "mxfp8_latency_ms": mxfp8_ms,
                "bf16_effective_tflops": total_flops / bf16_ms / 1e9,
                "mxfp8_effective_tflops": total_flops / mxfp8_ms / 1e9,
                "mxfp8_speedup_vs_bf16": bf16_ms / mxfp8_ms,
            },
            "full_model": {
                "layers": num_hidden_layers,
                "bf16_gemm_latency_ms": bf16_ms * num_hidden_layers,
                "mxfp8_gemm_latency_ms": mxfp8_ms * num_hidden_layers,
            },
        }

    def run_mode(mode: str) -> dict:
        entries = []
        for pass_name, shapes in shape_groups.items():
            for op_name, m, k, n in shapes:
                print(f"[{mode}] {pass_name}/{op_name}: {m}x{k}x{n}")
                flops = 2 * m * k * n
                bf16 = with_throughput(benchmark_bf16(m, k, n), flops)
                if mode == "autocast":
                    mxfp8_stats = benchmark_mxfp8_autocast(m, k, n)
                else:
                    mxfp8_stats = benchmark_mxfp8_prequantized(m, k, n)
                mxfp8 = with_throughput(mxfp8_stats, flops)
                entries.append(
                    {
                        "pass": pass_name,
                        "op": op_name,
                        "shape": {"m": m, "k": k, "n": n},
                        "flops": flops,
                        "bf16": bf16,
                        "mxfp8": mxfp8,
                        "mxfp8_speedup_vs_bf16": (
                            bf16["latency_ms"] / mxfp8["latency_ms"]
                        ),
                    }
                )
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
        return {
            "semantics": (
                "TE autocast; MXFP8 timing includes dynamic quantization"
                if mode == "autocast"
                else "pre-quantized MXFP8 inputs; raw GEMM timing"
            ),
            "entries": entries,
            "aggregate": aggregate(entries),
        }

    warm_gpu(gpu_warmup_seconds)
    torch.cuda.reset_peak_memory_stats()
    benchmark_started = time.perf_counter()
    modes = {"autocast": run_mode("autocast")}
    if include_prequantized:
        modes["prequantized"] = run_mode("prequantized")
    benchmark_seconds = time.perf_counter() - benchmark_started

    smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit,clocks.sm,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    result = {
        "status": "PASS",
        "run_id": run_id,
        "kernel": KERNEL_NAME,
        "model": {
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": num_attention_heads,
            "num_hidden_layers": num_hidden_layers,
            "micro_batch_size": micro_batch_size,
            "sequence_length": sequence_length,
            "tokens": tokens,
        },
        "precision": {
            "baseline": "BF16 torch.matmul",
            "mxfp8": "Transformer Engine MXFP8BlockScaling, E4M3 values, BF16 parameters/input/output",
        },
        "correctness": correctness,
        "benchmark": {
            "timer": "CUDA events",
            "gpu_warmup_seconds": gpu_warmup_seconds,
            "per_shape_warmup": warmup,
            "iterations_per_repeat": iterations,
            "repeats": repeats,
            "statistic": "median of repeat-average latencies",
            "benchmark_seconds": benchmark_seconds,
            "modes": modes,
        },
        "environment": {
            "gpu_name": gpu_name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": int(properties.total_memory),
            "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "transformer_engine": str(transformer_engine.__version__),
            "mxfp8_available": bool(is_available),
            "mxfp8_reason": str(reason),
            "nvidia_smi": smi,
        },
        "methodology": {
            "shape_source": "NVIDIA Transformer Engine benchmark_gemm.py 5B model example",
            "shape_source_url": "https://github.com/NVIDIA/TransformerEngine/blob/main/benchmarks/gemm/benchmark_gemm.py",
            "scope": "12 isolated linear GEMMs: 4 fprop, 4 dgrad, 4 wgrad",
            "not_included": "attention, normalization, activation, optimizer, communication",
        },
        "remote_function_seconds": time.perf_counter() - started,
        "remote_artifacts": {
            "volume": "cutex-autotune-cache",
            "run_directory": str(remote_run_dir),
        },
    }
    (remote_run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cache_volume.commit()
    return result


@app.local_entrypoint()
def probe():
    print(json.dumps(probe_remote.remote(), indent=2, ensure_ascii=False))


@app.local_entrypoint()
def main(
    hidden_size: int = 4096,
    intermediate_size: int = 16384,
    num_attention_heads: int = 32,
    num_hidden_layers: int = 24,
    micro_batch_size: int = 31,
    sequence_length: int = 512,
    warmup: int = 10,
    iterations: int = 100,
    repeats: int = 3,
    gpu_warmup_seconds: float = 5.0,
    include_prequantized: bool = True,
):
    result = run_remote.remote(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
        num_hidden_layers=num_hidden_layers,
        micro_batch_size=micro_batch_size,
        sequence_length=sequence_length,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        gpu_warmup_seconds=gpu_warmup_seconds,
        include_prequantized=include_prequantized,
    )
    run_dir = LOCAL_ARTIFACT_ROOT / f"{result['run_id']}-{result['kernel']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    compact = {
        "status": result["status"],
        "gpu": result["environment"]["gpu_name"],
        "transformer_engine": result["environment"]["transformer_engine"],
        "model": result["model"],
        "correctness": result["correctness"],
        "benchmark_seconds": result["benchmark"]["benchmark_seconds"],
        "remote_function_seconds": result["remote_function_seconds"],
        "modes": {
            name: mode["aggregate"]
            for name, mode in result["benchmark"]["modes"].items()
        },
        "local_artifact": str(result_path.resolve()),
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))
