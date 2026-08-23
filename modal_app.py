"""Modal entrypoint for remote CuTeDSL operator experiments."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-operator-lab"
CACHE_MOUNT = "/cutex-cache"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("cutex-autotune-cache", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])
    .uv_pip_install(
        "nvidia-cutlass-dsl[cu13]==4.7.0",
        "torch==2.13.0",
    )
    .env({"PYTHONUNBUFFERED": "1"})
    .add_local_python_source("cutex", copy=True)
)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


@app.function(
    image=image,
    gpu="L4",
    timeout=15 * 60,
    volumes={CACHE_MOUNT: cache_volume},
)
def run_vector_add_remote(
    m: int = 4096,
    n: int = 4096,
    warmup: int = 10,
    iterations: int = 100,
    force_retune: bool = False,
    trace: bool = False,
    dump_ir: bool = False,
) -> dict:
    import importlib.metadata
    import os

    import torch
    from cutlass.cute.runtime import from_dlpack

    import cutex
    from cutex.benchmark import cuda_benchmark
    from cutex.kernels.vector_add import vector_add
    from cutex.trace import disabled_iket_metadata, run_iket_profile

    if m < 1 or n < 1:
        raise ValueError("m and n must both be positive")
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    properties = torch.cuda.get_device_properties(0)
    if trace and properties.major < 9:
        raise RuntimeError(
            "CUTLASS IKET requires SM90 or newer; choose H100/H200/B200 "
            "or disable --trace"
        )
    gpu_name = str(properties.name)
    gpu_slug = _safe_slug(gpu_name)
    remote_run_dir = Path(CACHE_MOUNT) / "runs" / gpu_slug / run_id
    remote_run_dir.mkdir(parents=True, exist_ok=True)
    autotune_cache = Path(CACHE_MOUNT) / "autotune" / gpu_slug / "vector_add.json"

    if dump_ir:
        dump_dir = remote_run_dir / "cute-dsl-dump"
        dump_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CUTE_DSL_DUMP_DIR"] = str(dump_dir)
        os.environ["CUTE_DSL_KEEP_PTX"] = "1"
        os.environ["CUTE_DSL_LINEINFO"] = "1"

    torch.manual_seed(20260821)
    a = torch.randn((m, n), device="cuda", dtype=torch.float32)
    b = torch.randn((m, n), device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    a_cute = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
    b_cute = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
    c_cute = from_dlpack(c, assumed_align=16).mark_layout_dynamic()

    compiled = cutex.compile(
        vector_add,
        a_cute,
        b_cute,
        c_cute,
        m,
        n,
        verbose=True,
        force_retune=force_retune,
        cache_path=autotune_cache,
    )

    c.zero_()
    compiled(a_cute, b_cute, c_cute, m, n)
    torch.cuda.synchronize()

    expected = a + b
    torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-6)
    max_abs_error = float((c - expected).abs().max().item())

    stats = cuda_benchmark(
        compiled,
        a_cute,
        b_cute,
        c_cute,
        m,
        n,
        warmup=warmup,
        rep=iterations,
    )
    bytes_per_launch = 3 * a.numel() * a.element_size()
    bandwidth_gbps = bytes_per_launch / (stats.median_us * 1e-6) / 1e9
    trace_metadata = disabled_iket_metadata()
    if trace:
        copy_bits = int(compiled.best_config.kwargs["copy_bits"])
        trace_m = min(m, 512)
        trace_n = min(n, 512)
        try:
            iket_result = run_iket_profile(
                remote_run_dir / "iket",
                [
                    "vector-add",
                    "--m",
                    str(trace_m),
                    "--n",
                    str(trace_n),
                    "--copy-bits",
                    str(copy_bits),
                ],
            )
        except Exception:
            cache_volume.commit()
            raise
        trace_metadata = iket_result.to_metadata(volume_root=CACHE_MOUNT)
        trace_metadata["workload_shape"] = [trace_m, trace_n]

    result = {
        "status": "PASS",
        "run_id": run_id,
        "kernel": "vector_add",
        "shape": [m, n],
        "dtype": "float32",
        "correctness": {
            "max_abs_error": max_abs_error,
            "sample": c.flatten()[:8].tolist(),
        },
        "benchmark": {
            **stats.to_dict(include_samples=False),
            "instrumentation": "disabled",
            "warmup": warmup,
            "iterations": iterations,
            "bytes_per_launch": bytes_per_launch,
            "effective_bandwidth_gbps": bandwidth_gbps,
        },
        "autotune": compiled.metadata,
        "environment": {
            "gpu_name": gpu_name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": int(properties.total_memory),
            "torch": str(torch.__version__),
            "cutex": str(cutex.__version__),
            "nvidia_cutlass_dsl": str(
                importlib.metadata.version("nvidia-cutlass-dsl")
            ),
        },
        "remote_artifacts": {
            "volume": "cutex-autotune-cache",
            "run_directory": str(remote_run_dir),
            "autotune_cache": str(autotune_cache),
        },
        "trace": trace_metadata,
    }

    (remote_run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cache_volume.commit()
    return result


@app.local_entrypoint()
def main(
    gpu: str = "L4",
    m: int = 4096,
    n: int = 4096,
    warmup: int = 10,
    iterations: int = 100,
    force_retune: bool = False,
    trace: bool = False,
    dump_ir: bool = False,
):
    runner = run_vector_add_remote.with_options(gpu=gpu)
    result = runner.remote(
        m=m,
        n=n,
        warmup=warmup,
        iterations=iterations,
        force_retune=force_retune,
        trace=trace,
        dump_ir=dump_ir,
    )

    run_dir = LOCAL_ARTIFACT_ROOT / f"{result['run_id']}-{result['kernel']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    from cutex.trace import download_iket_artifacts

    trace_files = result["trace"].get("files", [])
    downloaded = download_iket_artifacts(cache_volume, trace_files, run_dir)
    result["trace"]["local_files"] = [str(path.resolve()) for path in downloaded]
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "status": result["status"],
        "gpu": result["environment"]["gpu_name"],
        "shape": result["shape"],
        "best_config": result["autotune"]["best_config"],
        "median_us": result["benchmark"]["median_us"],
        "effective_bandwidth_gbps": result["benchmark"]["effective_bandwidth_gbps"],
        "trace_backend": result["trace"]["backend"],
        "trace_files": result["trace"]["local_files"],
        "local_artifacts": str(run_dir.resolve()),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
