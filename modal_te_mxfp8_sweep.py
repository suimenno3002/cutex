"""Sweep rowwise-only MXFP8 Fprop GEMM sizes on one Modal B300."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-te-mxfp8-rowwise-fprop-sweep-b300"
CACHE_MOUNT = "/cutex-cache"
KERNEL_BASE_NAME = "te_mxfp8_rowwise_fprop_sweep_1024_32768"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
SIZES = (1024, 2048, 4096, 8192, 16384, 32768)
ROOFLINE_TFLOPS = 4500.0

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("cutex-autotune-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.07-py3")
    .entrypoint([])
    .env({"NVIDIA_IMEX_CHANNELS": "0", "PYTHONUNBUFFERED": "1"})
)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


@app.function(
    image=image,
    gpu="B300",
    timeout=30 * 60,
    volumes={CACHE_MOUNT: cache_volume},
)
def run_remote(
    warmup: int = 100,
    iterations: int = 500,
    gpu_warmup_seconds: float = 10.0,
    fast_accum: bool = False,
) -> dict:
    import math
    import statistics
    import subprocess
    import time

    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    if warmup < 0 or iterations < 1 or gpu_warmup_seconds < 0:
        raise ValueError("warmup must be >= 0, iterations >= 1, GPU warmup >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")
    availability = te.is_mxfp8_available(return_reason=True)
    available, reason = (
        availability if isinstance(availability, tuple) else (bool(availability), "")
    )
    if not available:
        raise RuntimeError(f"Transformer Engine MXFP8 is unavailable: {reason}")
    kernel_name = f"{KERNEL_BASE_NAME}_{'fast_accum' if fast_accum else 'split_accum'}"

    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (10, 3):
        raise RuntimeError(
            f"this sweep targets B300/SM103; got {properties.name} "
            f"(SM{properties.major}{properties.minor})"
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    remote_run_dir = (
        Path(CACHE_MOUNT)
        / "runs"
        / _safe_slug(str(properties.name))
        / run_id
        / kernel_name
    )
    remote_run_dir.mkdir(parents=True, exist_ok=True)

    def warm_gpu(duration_seconds: float) -> None:
        if duration_seconds <= 0:
            return
        lhs = torch.randn((4096, 4096), device="cuda", dtype=torch.bfloat16)
        rhs = torch.randn_like(lhs)
        torch.cuda.synchronize()
        until = time.perf_counter() + duration_seconds
        while time.perf_counter() < until:
            for _ in range(10):
                torch.mm(lhs, rhs)
            torch.cuda.synchronize()
        del lhs, rhs
        torch.cuda.empty_cache()

    def benchmark(fn) -> dict:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        pairs = []
        for _ in range(iterations):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            pairs.append((start, end))
        pairs[-1][1].synchronize()
        samples = [start.elapsed_time(end) * 1000.0 for start, end in pairs]
        ordered = sorted(samples)
        return {
            "mean_us": statistics.fmean(ordered),
            "median_us": statistics.median(ordered),
            "min_us": ordered[0],
            "max_us": ordered[-1],
            "p95_us": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        }

    warm_gpu(gpu_warmup_seconds)
    points = []
    for size in SIZES:
        torch.manual_seed(20260822 + size)
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn((size, size), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn_like(x)

        def quantize_rowwise(tensor):
            quantizer = te.MXFP8Quantizer(
                tex.DType.kFloat8E4M3, rowwise=True, columnwise=False
            )
            quantizer.optimize_for_gemm = True
            return quantizer.quantize(tensor)

        x_q, weight_q = quantize_rowwise(x), quantize_rowwise(weight)
        for name, tensor in (("x", x_q), ("weight", weight_q)):
            usages = tensor.get_usages()
            if not usages["rowwise"] or usages["columnwise"]:
                raise AssertionError(f"{name} is not rowwise-only MXFP8")
            if not tensor._with_gemm_swizzled_scales:
                raise AssertionError(f"{name} scales are not GEMM-swizzled")

        output = torch.empty_like(x)

        def fprop() -> None:
            general_gemm(
                weight_q,
                x_q,
                out_dtype=torch.bfloat16,
                out=output,
                layout="TN",
                use_split_accumulator=not fast_accum,
            )

        fprop()
        torch.cuda.synchronize()
        reference = torch.mm(x.float(), weight.float().T)
        difference = output.float() - reference
        relative_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference)
        )
        correctness = {
            "all_finite": bool(torch.isfinite(output).all().item()),
            "relative_l2_vs_fp32": relative_l2,
            "max_abs_vs_fp32": float(difference.abs().max().item()),
        }
        if not correctness["all_finite"] or relative_l2 >= 0.15:
            raise AssertionError(f"size={size} correctness failed: {correctness}")
        del reference, difference

        stats = benchmark(fprop)
        flop_count = 2 * size**3
        tflops = flop_count / stats["median_us"] / 1e6
        points.append(
            {
                "size": size,
                "shape": {"m": size, "n": size, "k": size},
                "flop_count": flop_count,
                **stats,
                "tflops": tflops,
                "roofline_tflops": ROOFLINE_TFLOPS,
                "roofline_utilization_percent": 100.0 * tflops / ROOFLINE_TFLOPS,
                "correctness": correctness,
                "peak_allocated_memory_bytes": int(torch.cuda.max_memory_allocated()),
            }
        )
        del x, weight, x_q, weight_q, output
        torch.cuda.empty_cache()

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
        "kernel": kernel_name,
        "operation": "Y[M,N] = X[M,K] @ W[N,K].T",
        "precision": {
            "inputs": "rowwise-only MXFP8 E4M3, one E8M0 scale per 32 K values",
            "multiply": "FP8 E4M3 x E4M3",
            "compute_type": "CUBLAS_COMPUTE_32F (fixed by Transformer Engine)",
            "use_split_accumulator": not fast_accum,
            "fast_accumulation_requested": fast_accum,
            "accumulation": (
                "FP32 compute contract; FP8 fast accumulation requested"
                if fast_accum
                else "FP32 compute contract; split accumulation enabled"
            ),
            "output": "BF16",
            "quantization_in_timed_region": False,
        },
        "benchmark": {
            "timer": "CUDA events",
            "gpu_warmup_seconds": gpu_warmup_seconds,
            "warmup_per_shape": warmup,
            "iterations_per_shape": iterations,
            "statistic": "median of individual kernel launches",
        },
        "points": points,
        "environment": {
            "gpu_name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": int(properties.total_memory),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "transformer_engine": str(transformer_engine.__version__),
            "mxfp8_available": bool(available),
            "mxfp8_reason": str(reason),
            "nvidia_smi": smi,
        },
        "remote_artifacts": {
            "volume": "cutex-autotune-cache",
            "run_directory": str(remote_run_dir),
        },
    }
    (remote_run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    cache_volume.commit()
    return result


@app.local_entrypoint()
def main(
    warmup: int = 100,
    iterations: int = 500,
    gpu_warmup_seconds: float = 10.0,
    fast_accum: bool = False,
):
    result = run_remote.remote(
        warmup=warmup,
        iterations=iterations,
        gpu_warmup_seconds=gpu_warmup_seconds,
        fast_accum=fast_accum,
    )
    run_dir = LOCAL_ARTIFACT_ROOT / f"{result['run_id']}-{result['kernel']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "precision": result["precision"],
                "benchmark": result["benchmark"],
                "points": result["points"],
                "environment": result["environment"],
                "local_artifact": str(result_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
