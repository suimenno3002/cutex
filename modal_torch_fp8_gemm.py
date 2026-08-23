"""Benchmark PyTorch tensor-wise FP8 GEMM on one Modal B300."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-torch-fp8-gemm"
CACHE_MOUNT = "/cutex-cache"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
B300_DENSE_FP8_TFLOPS = 4_500.0

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("cutex-autotune-cache", create_if_missing=True)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.2.1-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "torch==2.13.0",
        index_url="https://download.pytorch.org/whl/cu132",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("cutex", copy=True)
)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


@app.function(
    image=image,
    gpu="B300",
    timeout=15 * 60,
    volumes={CACHE_MOUNT: cache_volume},
)
def run_torch_fp8_gemm_remote(
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    warmup: int = 1000,
    iterations: int = 500,
) -> dict:
    import inspect
    import subprocess
    import time

    import torch
    import torch.nn.functional as F

    from cutex.benchmark import cuda_benchmark

    started = time.perf_counter()
    if min(m, n, k) <= 0 or any(dim % 16 for dim in (m, n, k)):
        raise ValueError("m, n, and k must be positive multiples of 16")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")
    if not hasattr(F, "scaled_mm") or not hasattr(F, "ScalingType"):
        raise RuntimeError(f"PyTorch {torch.__version__} lacks public scaled_mm")

    properties = torch.cuda.get_device_properties(0)
    gpu_name = str(properties.name)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    remote_run_dir = (
        Path(CACHE_MOUNT)
        / "runs"
        / _safe_slug(gpu_name)
        / run_id
        / "pytorch_fp8_gemm"
    )
    remote_run_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260822)
    fp8 = torch.float8_e4m3fn
    a = (torch.randn((m, k), device="cuda") * 0.125).to(fp8)
    b_storage = (torch.randn((n, k), device="cuda") * 0.125).to(fp8)
    b = b_storage.transpose(0, 1)
    scale_a = torch.ones((), device="cuda", dtype=torch.float32)
    scale_b = torch.ones((), device="cuda", dtype=torch.float32)
    scaling = F.ScalingType.TensorWise

    def scaled_mm(use_fast_accum: bool):
        return F.scaled_mm(
            a,
            b,
            scale_a,
            scaling,
            scale_b,
            scaling,
            output_dtype=torch.bfloat16,
            use_fast_accum=use_fast_accum,
        )

    outputs = {
        "fast_accum_false": scaled_mm(False),
        "fast_accum_true": scaled_mm(True),
    }
    reference = torch.mm(a.float(), b.float())
    torch.cuda.synchronize()

    correctness = {}
    for name, output in outputs.items():
        difference = output.float() - reference
        relative_l2 = float(
            torch.linalg.vector_norm(difference)
            / torch.linalg.vector_norm(reference)
        )
        correctness[name] = {
            "all_finite": bool(torch.isfinite(output).all().item()),
            "max_abs_error": float(difference.abs().max().item()),
            "mean_abs_error": float(difference.abs().mean().item()),
            "relative_l2_error": relative_l2,
        }
        if not correctness[name]["all_finite"] or relative_l2 >= 0.1:
            raise AssertionError(f"{name} failed correctness: {correctness[name]}")

    flop_count = 2 * m * n * k
    benchmark = {}
    for use_fast_accum in (False, True):
        name = f"fast_accum_{str(use_fast_accum).lower()}"
        stats = cuda_benchmark(
            scaled_mm,
            use_fast_accum,
            warmup=warmup,
            rep=iterations,
            stream=torch.cuda.current_stream(),
        )
        tflops = flop_count / stats.median_us / 1e6
        benchmark[name] = {
            **stats.to_dict(include_samples=True),
            "tflops": tflops,
            "percent_of_b300_dense_fp8_peak": (
                100.0 * tflops / B300_DENSE_FP8_TFLOPS
            ),
        }

    def nvidia_smi() -> str:
        command = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,power.limit,clocks.sm,clocks.max.sm",
            "--format=csv,noheader,nounits",
        ]
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()

    result = {
        "status": "PASS",
        "run_id": run_id,
        "kernel": "pytorch_fp8_gemm",
        "operation": "C[M,N] = A[M,K] @ B[K,N]",
        "shape": {"m": m, "n": n, "k": k},
        "dtype": {
            "a": "float8_e4m3fn",
            "b": "float8_e4m3fn",
            "output": "bfloat16",
        },
        "layout": {
            "a_shape": list(a.shape),
            "a_stride": list(a.stride()),
            "b_shape": list(b.shape),
            "b_stride": list(b.stride()),
        },
        "scaling": {
            "a": "TensorWise",
            "b": "TensorWise",
            "scale_a": 1.0,
            "scale_b": 1.0,
        },
        "correctness": {
            "reference": "torch.mm(a.float(), b.float())",
            **correctness,
        },
        "benchmark": {
            "timer": "CUDA events",
            "warmup": warmup,
            "iterations": iterations,
            "flop_count": flop_count,
            "nominal_b300_dense_fp8_tflops": B300_DENSE_FP8_TFLOPS,
            **benchmark,
        },
        "environment": {
            "gpu_name": gpu_name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": int(properties.total_memory),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "cudnn": int(torch.backends.cudnn.version()),
            "scaled_mm_signature": str(inspect.signature(F.scaled_mm)),
            "nvidia_smi": nvidia_smi(),
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
    m: int = 4096,
    n: int = 4096,
    k: int = 4096,
    warmup: int = 1000,
    iterations: int = 500,
):
    result = run_torch_fp8_gemm_remote.remote(
        m=m,
        n=n,
        k=k,
        warmup=warmup,
        iterations=iterations,
    )
    run_dir = LOCAL_ARTIFACT_ROOT / f"{result['run_id']}-{result['kernel']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    compact = {}
    for name in ("fast_accum_false", "fast_accum_true"):
        compact[name] = {
            key: value
            for key, value in result["benchmark"][name].items()
            if key != "samples_us"
        }
    summary = {
        "status": result["status"],
        "gpu": result["environment"]["gpu_name"],
        "shape": result["shape"],
        "torch": result["environment"]["torch"],
        "torch_cuda": result["environment"]["torch_cuda"],
        **compact,
        "remote_function_seconds": result["remote_function_seconds"],
        "local_artifacts": str(run_dir.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
