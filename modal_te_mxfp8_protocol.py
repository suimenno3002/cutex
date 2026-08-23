"""Measure rowwise MXFP8 GEMM on B300 with a power-aware launch protocol."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-te-mxfp8-protocol-b300"
CACHE_MOUNT = "/cutex-cache"
KERNEL_BASE_NAME = "te_mxfp8_rowwise_fprop"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
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
    size: int = 4096,
    repeats: int = 10,
    launch_interval_ms: int = 250,
    per_sample_warmup: int = 2,
    sample_interval_ms: int = 5,
) -> dict:
    import csv
    import statistics
    import subprocess
    import threading
    import time

    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    started = time.perf_counter()
    if size <= 0 or size % 32:
        raise ValueError("size must be a positive multiple of 32")
    if repeats < 1 or launch_interval_ms < 0 or per_sample_warmup < 0:
        raise ValueError("invalid benchmark protocol")
    if sample_interval_ms < 5:
        raise ValueError("sample_interval_ms must be >= 5")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")

    availability = te.is_mxfp8_available(return_reason=True)
    is_available, reason = (
        availability if isinstance(availability, tuple) else (bool(availability), "")
    )
    if not is_available:
        raise RuntimeError(f"Transformer Engine MXFP8 is unavailable: {reason}")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (10, 3):
        raise RuntimeError(
            f"this benchmark targets B300/SM103; got {properties.name} "
            f"(SM{properties.major}{properties.minor})"
        )

    kernel_name = (
        f"{KERNEL_BASE_NAME}_{size}_spaced_{launch_interval_ms}ms_"
        f"random_warmup{per_sample_warmup}"
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

    telemetry_fields = (
        ("timestamp", "timestamp", False),
        ("power_w", "power.draw", True),
        ("power_limit_w", "power.limit", True),
        ("sm_clock_mhz", "clocks.sm", True),
        ("max_sm_clock_mhz", "clocks.max.sm", True),
        ("gpu_util_percent", "utilization.gpu", True),
        ("temperature_c", "temperature.gpu", True),
        ("pstate", "pstate", False),
        ("sw_power_cap", "clocks_event_reasons.sw_power_cap", False),
        (
            "hw_thermal_slowdown",
            "clocks_event_reasons.hw_thermal_slowdown",
            False,
        ),
    )
    smi_query = ",".join(field for _name, field, _numeric in telemetry_fields)

    def parse_value(value: str, numeric: bool):
        value = value.strip()
        if not numeric:
            return value
        try:
            return float(value)
        except ValueError:
            return None

    def start_monitor():
        process = subprocess.Popen(
            [
                "nvidia-smi",
                f"--query-gpu={smi_query}",
                "--format=csv,noheader,nounits",
                f"--loop-ms={sample_interval_ms}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        monitor_started = time.perf_counter()
        samples = []

        def read_samples():
            assert process.stdout is not None
            for line in process.stdout:
                values = next(csv.reader([line]))
                if len(values) != len(telemetry_fields):
                    continue
                sample = {"host_time_s": time.perf_counter() - monitor_started}
                for (name, _field, numeric), value in zip(telemetry_fields, values):
                    sample[name] = parse_value(value, numeric)
                samples.append(sample)

        reader = threading.Thread(target=read_samples, daemon=True)
        reader.start()
        return process, reader, monitor_started, samples

    def stop_monitor(process, reader):
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        reader.join(timeout=5)
        return process.stderr.read().strip() if process.stderr is not None else ""

    def numeric_stats(samples, key: str):
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        if not values:
            return None
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }

    torch.manual_seed(20260822)
    dtype = torch.bfloat16
    x = torch.randn((size, size), device="cuda", dtype=dtype)
    weight = torch.randn((size, size), device="cuda", dtype=dtype)

    def quantize_rowwise(tensor):
        quantizer = te.MXFP8Quantizer(
            tex.DType.kFloat8E4M3, rowwise=True, columnwise=False
        )
        quantizer.optimize_for_gemm = True
        quantized = quantizer.quantize(tensor)
        usages = quantized.get_usages()
        if not usages["rowwise"] or usages["columnwise"]:
            raise AssertionError("input is not rowwise-only MXFP8")
        if not quantized._with_gemm_swizzled_scales:
            raise AssertionError("MXFP8 scales are not GEMM-swizzled")
        return quantized

    x_q = quantize_rowwise(x)
    weight_q = quantize_rowwise(weight)
    output = torch.empty((size, size), device="cuda", dtype=dtype)

    def fprop() -> None:
        general_gemm(
            weight_q,
            x_q,
            out_dtype=dtype,
            out=output,
            layout="TN",
            use_split_accumulator=True,
        )

    # This call pays compilation/setup cost and validates output, outside the protocol.
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

    flop_count = 2 * size**3
    process, reader, monitor_started, samples = start_monitor()
    benchmark_start_s = time.perf_counter() - monitor_started
    sample_results = []
    for sample_index in range(repeats):
        wait_start_s = time.perf_counter() - monitor_started
        time.sleep(launch_interval_ms / 1000.0)
        warmup_start_s = time.perf_counter() - monitor_started
        for _ in range(per_sample_warmup):
            fprop()
        warmup_end_s = time.perf_counter() - monitor_started

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        measure_start_s = time.perf_counter() - monitor_started
        start_event.record()
        fprop()
        end_event.record()
        end_event.synchronize()
        measure_end_s = time.perf_counter() - monitor_started
        latency_us = float(start_event.elapsed_time(end_event)) * 1000.0
        sample_results.append(
            {
                "sample_index": sample_index,
                "latency_us": latency_us,
                "tflops": flop_count / latency_us / 1e6,
                "roofline_utilization_percent": (
                    100.0 * flop_count / latency_us / 1e6 / ROOFLINE_TFLOPS
                ),
                "wait_start_s": wait_start_s,
                "warmup_start_s": warmup_start_s,
                "warmup_end_s": warmup_end_s,
                "measure_start_s": measure_start_s,
                "measure_end_s": measure_end_s,
            }
        )
        print(
            f"sample={sample_index} latency_us={latency_us:.3f} "
            f"tflops={sample_results[-1]['tflops']:.2f}"
        )
    benchmark_end_s = time.perf_counter() - monitor_started
    time.sleep(0.5)
    monitor_stderr = stop_monitor(process, reader)

    benchmark_samples = [
        sample
        for sample in samples
        if benchmark_start_s <= sample["host_time_s"] <= benchmark_end_s
    ]
    if len(benchmark_samples) < 2:
        raise RuntimeError(f"too few telemetry samples: {monitor_stderr}")
    active_count = sum(
        sample.get("sw_power_cap") == "Active" for sample in benchmark_samples
    )
    latencies = [sample["latency_us"] for sample in sample_results]
    tflops_samples = [sample["tflops"] for sample in sample_results]
    median_latency_us = statistics.median(latencies)
    benchmark = {
        "scope": "pre-quantized rowwise-only MXFP8 Fprop GEMM",
        "timer": "CUDA events",
        "repeats": repeats,
        "flop_count": flop_count,
        "latency_us": median_latency_us,
        "mean_latency_us": statistics.fmean(latencies),
        "min_latency_us": min(latencies),
        "max_latency_us": max(latencies),
        "tflops": flop_count / median_latency_us / 1e6,
        "mean_sample_tflops": statistics.fmean(tflops_samples),
        "min_sample_tflops": min(tflops_samples),
        "max_sample_tflops": max(tflops_samples),
        "roofline_tflops": ROOFLINE_TFLOPS,
        "roofline_utilization_percent": (
            100.0 * flop_count / median_latency_us / 1e6 / ROOFLINE_TFLOPS
        ),
        "samples": sample_results,
    }
    telemetry = {
        "scope": "wall interval including launch spacing and unmeasured warmups",
        "sample_interval_ms_requested": sample_interval_ms,
        "sample_count": len(benchmark_samples),
        "power_w": numeric_stats(benchmark_samples, "power_w"),
        "power_limit_w": numeric_stats(benchmark_samples, "power_limit_w"),
        "sm_clock_mhz": numeric_stats(benchmark_samples, "sm_clock_mhz"),
        "gpu_util_percent": numeric_stats(benchmark_samples, "gpu_util_percent"),
        "temperature_c": numeric_stats(benchmark_samples, "temperature_c"),
        "sw_power_cap_active_count": active_count,
        "sw_power_cap_active_percent": 100.0 * active_count / len(benchmark_samples),
        "sw_power_cap_values": sorted(
            {sample.get("sw_power_cap") for sample in benchmark_samples}
        ),
        "hw_thermal_slowdown_values": sorted(
            {sample.get("hw_thermal_slowdown") for sample in benchmark_samples}
        ),
        "benchmark_start_s": benchmark_start_s,
        "benchmark_end_s": benchmark_end_s,
        "monitor_stderr": monitor_stderr,
        "samples": samples,
    }

    static_smi = subprocess.run(
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
        "operation": {
            "pass": "Fprop",
            "formula": "Y = X @ W.T",
            "shape": {"m": size, "n": size, "k": size},
            "te_call": "general_gemm(weight_q, x_q, layout='TN')",
        },
        "precision": {
            "inputs": "MXFP8 E4M3, rowwise-only, one E8M0 scale per 32 K values",
            "multiply": "FP8 E4M3 x E4M3",
            "compute_type": "CUBLAS_COMPUTE_32F",
            "accumulation": "FP32 contract; fast accumulation explicitly disabled",
            "use_split_accumulator": True,
            "output": "bfloat16",
            "quantization_in_timed_region": False,
        },
        "methodology": {
            "protocol": (
                "wait, run unmeasured random-input warmups, then time one GEMM"
            ),
            "launch_interval_ms": launch_interval_ms,
            "per_sample_warmup": per_sample_warmup,
            "measured_gemms_per_sample": 1,
            "repeats": repeats,
            "statistic": f"median of {repeats} single-GEMM CUDA-event latencies",
            "compile_and_validation_calls_in_timed_region": False,
            "continuous_global_warmup": False,
        },
        "correctness": correctness,
        "benchmark": benchmark,
        "telemetry": telemetry,
        "environment": {
            "gpu_name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": int(properties.total_memory),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "transformer_engine": str(transformer_engine.__version__),
            "mxfp8_available": bool(is_available),
            "mxfp8_reason": str(reason),
            "nvidia_smi": static_smi,
        },
        "remote_function_seconds": time.perf_counter() - started,
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
    size: int = 4096,
    repeats: int = 10,
    launch_interval_ms: int = 250,
    per_sample_warmup: int = 2,
    sample_interval_ms: int = 5,
):
    result = run_remote.remote(
        size=size,
        repeats=repeats,
        launch_interval_ms=launch_interval_ms,
        per_sample_warmup=per_sample_warmup,
        sample_interval_ms=sample_interval_ms,
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
                "operation": result["operation"],
                "precision": result["precision"],
                "methodology": result["methodology"],
                "correctness": result["correctness"],
                "benchmark": result["benchmark"],
                "telemetry": {
                    key: value
                    for key, value in result["telemetry"].items()
                    if key != "samples"
                },
                "environment": result["environment"],
                "local_artifact": str(result_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
