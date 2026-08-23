"""Measure one rowwise-only MXFP8 Fprop GEMM on a Modal B300."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-te-mxfp8-rowwise-fprop-4096"
CACHE_MOUNT = "/cutex-cache"
KERNEL_NAME = "te_mxfp8_rowwise_fprop_4096"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"

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
    size: int = 4096,
    warmup: int = 200,
    iterations: int = 1000,
    repeats: int = 10,
    gpu_warmup_seconds: float = 10.0,
) -> dict:
    import statistics
    import subprocess
    import time

    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")
    if size <= 0 or size % 32:
        raise ValueError("size must be a positive multiple of 32")
    if warmup < 0 or iterations < 1 or repeats < 1 or gpu_warmup_seconds < 0:
        raise ValueError("invalid benchmark iteration or warmup setting")

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

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(20260822)

    def warm_gpu(duration_seconds: float) -> None:
        if duration_seconds <= 0:
            return
        a = torch.randn((4096, 4096), device=device, dtype=dtype)
        b = torch.randn((4096, 4096), device=device, dtype=dtype)
        torch.cuda.synchronize()
        until = time.perf_counter() + duration_seconds
        while time.perf_counter() < until:
            for _ in range(10):
                torch.mm(a, b)
            torch.cuda.synchronize()
        del a, b
        torch.cuda.empty_cache()

    def time_cuda(run_fn) -> dict:
        for _ in range(warmup):
            run_fn()
        torch.cuda.synchronize()
        samples_us = []
        for _ in range(repeats):
            run_fn()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iterations):
                run_fn()
            end_event.record()
            end_event.synchronize()
            samples_us.append(
                float(start_event.elapsed_time(end_event)) * 1000.0 / iterations
            )
        ordered = sorted(samples_us)
        latency_us = statistics.median(ordered)
        return {
            "latency_us": latency_us,
            "mean_latency_us": statistics.fmean(ordered),
            "min_latency_us": ordered[0],
            "max_latency_us": ordered[-1],
            "repeat_average_latency_us": samples_us,
        }

    warm_gpu(gpu_warmup_seconds)
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn((size, size), device=device, dtype=dtype)
    weight = torch.randn((size, size), device=device, dtype=dtype)

    def rowwise_quantizer():
        quantizer = te.MXFP8Quantizer(
            tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        quantizer.optimize_for_gemm = True
        return quantizer

    x_q = rowwise_quantizer().quantize(x)
    weight_q = rowwise_quantizer().quantize(weight)
    input_storage = {}
    for name, tensor in (("x", x_q), ("weight", weight_q)):
        usages = tensor.get_usages()
        input_storage[name] = {
            "rowwise": bool(usages["rowwise"]),
            "columnwise": bool(usages["columnwise"]),
            "rowwise_data_present": tensor._rowwise_data is not None,
            "columnwise_data_present": tensor._columnwise_data is not None,
            "gemm_swizzled_scales": bool(tensor._with_gemm_swizzled_scales),
        }
        if input_storage[name]["rowwise"] is not True:
            raise AssertionError(f"{name} is missing rowwise MXFP8 data")
        if input_storage[name]["columnwise"] is not False:
            raise AssertionError(f"{name} unexpectedly contains columnwise MXFP8 data")

    output = torch.empty((size, size), device=device, dtype=dtype)

    def fprop() -> None:
        general_gemm(
            weight_q,
            x_q,
            out_dtype=dtype,
            out=output,
            layout="TN",
            use_split_accumulator=True,
        )

    fprop()
    torch.cuda.synchronize()
    reference = torch.mm(x.float(), weight.float().T)
    difference = output.float() - reference
    relative_l2 = float(
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
    )
    correctness = {
        "all_finite": bool(torch.isfinite(output).all().item()),
        "output_dtype": str(output.dtype),
        "relative_l2_error_vs_fp32": relative_l2,
        "max_abs_error_vs_fp32": float(difference.abs().max().item()),
    }
    if not correctness["all_finite"] or relative_l2 >= 0.15:
        raise AssertionError(f"Fprop correctness failed: {correctness}")
    del reference, difference

    benchmark_started = time.perf_counter()
    stats = time_cuda(fprop)
    benchmark_seconds = time.perf_counter() - benchmark_started
    flop_count = 2 * size**3
    stats["tflops"] = flop_count / stats["latency_us"] / 1e6

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
        "operation": {
            "pass": "Fprop",
            "formula": "Y = X @ W.T",
            "shape": {"m": size, "n": size, "k": size},
            "te_call": "general_gemm(weight_q, x_q, layout='TN')",
            "te_layout": "TN",
        },
        "precision": {
            "inputs": "MXFP8 E4M3, rowwise-only, one E8M0 scale per 32 values",
            "input_storage": input_storage,
            "multiply": "FP8 E4M3 x E4M3",
            "compute_type": "CUBLAS_COMPUTE_32F",
            "scale_type": "CUDA_R_32F",
            "accumulation": "FP32 contract; fast accumulation explicitly disabled",
            "use_split_accumulator": True,
            "output": "bfloat16",
            "quantization_in_timed_region": False,
        },
        "correctness": correctness,
        "benchmark": {
            "scope": "pre-quantized rowwise-only MXFP8 Fprop GEMM",
            "timer": "CUDA events",
            "gpu_warmup_seconds": gpu_warmup_seconds,
            "warmup_iterations": warmup,
            "iterations_per_repeat": iterations,
            "repeats": repeats,
            "statistic": "median of repeat-average latencies",
            "benchmark_seconds": benchmark_seconds,
            "flop_count": flop_count,
            **stats,
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
def main(
    size: int = 4096,
    warmup: int = 200,
    iterations: int = 1000,
    repeats: int = 10,
    gpu_warmup_seconds: float = 10.0,
):
    result = run_remote.remote(
        size=size,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        gpu_warmup_seconds=gpu_warmup_seconds,
    )
    run_dir = LOCAL_ARTIFACT_ROOT / f"{result['run_id']}-{result['kernel']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "operation": result["operation"],
                "precision": result["precision"],
                "correctness": result["correctness"],
                "benchmark": result["benchmark"],
                "environment": result["environment"],
                "local_artifact": str(result_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
