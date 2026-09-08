"""Compare BF16 and MXFP8 GEMM throughput on one Modal B300.

The public inputs and output are BF16 for both paths.  The MXFP8 path is
reported twice:

* ``mxfp8_core`` quantizes BF16 inputs before timing, so it measures the raw
  block-scaled GEMM throughput.
* ``mxfp8_end_to_end`` includes BF16 -> MXFP8 quantization in the timed region,
  so it measures the effective throughput seen at a BF16 API boundary.

Run the default peak sweep with:

    uv run modal run modal_b300_mxfp8_vs_bf16.py
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-b300-mxfp8-vs-bf16"
CACHE_MOUNT = "/cutex-cache"
KERNEL_NAME = "b300_mxfp8_vs_bf16_peak"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
DEFAULT_SIZES = (4096, 8192, 16384, 32768)
DEFAULT_INPUT_MODES = ("random", "ones")

# NVIDIA quotes sparse Tensor Core throughput for DGX B300.  Dividing by two
# for dense math and by eight GPUs gives these per-GPU reference rooflines.
BF16_DENSE_ROOFLINE_TFLOPS = 2250.0
MXFP8_DENSE_ROOFLINE_TFLOPS = 4500.0
ROOFLINE_SOURCE = "https://www.nvidia.com/en-us/data-center/dgx-b300/"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("cutex-autotune-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.07-py3")
    .entrypoint([])
    .env({"NVIDIA_IMEX_CHANNELS": "0", "PYTHONUNBUFFERED": "1"})
)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("at least one GEMM size is required")
    return parsed


def _parse_csv_strings(value: str) -> tuple[str, ...]:
    parsed = tuple(item.strip().lower() for item in value.split(",") if item.strip())
    if not parsed:
        raise ValueError("at least one input mode is required")
    return parsed


@app.function(
    image=image,
    gpu="B300",
    timeout=30 * 60,
    volumes={CACHE_MOUNT: cache_volume},
)
def run_remote(
    sizes: tuple[int, ...] = DEFAULT_SIZES,
    input_modes: tuple[str, ...] = DEFAULT_INPUT_MODES,
    warmup: int = 20,
    iterations: int = 20,
    repeats: int = 5,
    gpu_warmup_seconds: float = 10.0,
    telemetry_interval_ms: int = 20,
    include_end_to_end: bool = True,
    fast_accum: bool = True,
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

    sizes = tuple(int(size) for size in sizes)
    input_modes = tuple(str(mode).lower() for mode in input_modes)
    if any(size <= 0 or size % 32 for size in sizes):
        raise ValueError("all sizes must be positive multiples of 32")
    if any(mode not in {"random", "ones"} for mode in input_modes):
        raise ValueError("input modes must be 'random' and/or 'ones'")
    if warmup < 0 or iterations < 1 or repeats < 1:
        raise ValueError("warmup must be >= 0; iterations and repeats must be >= 1")
    if gpu_warmup_seconds < 0 or telemetry_interval_ms < 5:
        raise ValueError("invalid GPU warmup or telemetry interval")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")

    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (10, 3):
        raise RuntimeError(
            f"expected B300/SM103, got {properties.name} "
            f"(SM{properties.major}{properties.minor})"
        )
    availability = te.is_mxfp8_available(return_reason=True)
    mxfp8_available, mxfp8_reason = (
        availability if isinstance(availability, tuple) else (bool(availability), "")
    )
    if not mxfp8_available:
        raise RuntimeError(f"Transformer Engine MXFP8 is unavailable: {mxfp8_reason}")

    device = torch.device("cuda")
    torch.manual_seed(20260907)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False

    def new_quantizer():
        quantizer = te.MXFP8Quantizer(
            tex.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        quantizer.optimize_for_gemm = True
        return quantizer

    def quantize_rowwise(quantizer, tensor, name: str):
        quantized = quantizer.quantize(tensor)
        usages = quantized.get_usages()
        if not usages["rowwise"] or usages["columnwise"]:
            raise AssertionError(f"{name} is not rowwise-only MXFP8")
        if not quantized._with_gemm_swizzled_scales:
            raise AssertionError(f"{name} scales are not GEMM-swizzled")
        return quantized

    def launch_mxfp8(weight_q, x_q, output) -> None:
        general_gemm(
            weight_q,
            x_q,
            out_dtype=torch.bfloat16,
            out=output,
            layout="TN",
            use_split_accumulator=not fast_accum,
        )

    def error_metrics(output, reference) -> dict:
        difference = output.float() - reference
        reference_norm = torch.linalg.vector_norm(reference)
        return {
            "all_finite": bool(torch.isfinite(output).all().item()),
            "relative_l2_vs_fp32": float(
                torch.linalg.vector_norm(difference) / reference_norm
            ),
            "max_abs_vs_fp32": float(difference.abs().max().item()),
        }

    # A small independent check establishes the actual BF16 -> MXFP8 -> BF16
    # numerical path without adding a full FP32 reference GEMM to every peak point.
    check_size = 1024
    check_x = torch.empty(
        (check_size, check_size), device=device, dtype=torch.bfloat16
    ).uniform_(-1.0, 1.0)
    check_weight = torch.empty_like(check_x).uniform_(-1.0, 1.0)
    check_reference = torch.mm(check_x.float(), check_weight.float().T)
    check_bf16 = torch.empty_like(check_x)
    torch.mm(check_x, check_weight.T, out=check_bf16)
    check_x_quantizer = new_quantizer()
    check_weight_quantizer = new_quantizer()
    check_x_q = quantize_rowwise(check_x_quantizer, check_x, "check_x")
    check_weight_q = quantize_rowwise(
        check_weight_quantizer, check_weight, "check_weight"
    )
    check_mxfp8 = torch.empty_like(check_x)
    launch_mxfp8(check_weight_q, check_x_q, check_mxfp8)
    check_e2e_x_quantizer = new_quantizer()
    check_e2e_weight_quantizer = new_quantizer()
    check_e2e_x_q = check_e2e_x_quantizer.quantize(check_x)
    check_e2e_weight_q = check_e2e_weight_quantizer.quantize(check_weight)
    check_mxfp8_end_to_end = torch.empty_like(check_x)
    launch_mxfp8(check_e2e_weight_q, check_e2e_x_q, check_mxfp8_end_to_end)
    torch.cuda.synchronize()
    correctness = {
        "shape": {"m": check_size, "n": check_size, "k": check_size},
        "bf16": error_metrics(check_bf16, check_reference),
        "mxfp8": error_metrics(check_mxfp8, check_reference),
        "mxfp8_end_to_end": error_metrics(
            check_mxfp8_end_to_end, check_reference
        ),
    }
    if (
        not correctness["bf16"]["all_finite"]
        or correctness["bf16"]["relative_l2_vs_fp32"] >= 0.02
        or not correctness["mxfp8"]["all_finite"]
        or correctness["mxfp8"]["relative_l2_vs_fp32"] >= 0.15
        or not correctness["mxfp8_end_to_end"]["all_finite"]
        or correctness["mxfp8_end_to_end"]["relative_l2_vs_fp32"] >= 0.15
    ):
        raise AssertionError(f"correctness check failed: {correctness}")
    del (
        check_x,
        check_weight,
        check_reference,
        check_bf16,
        check_x_q,
        check_weight_q,
        check_mxfp8,
        check_e2e_x_q,
        check_e2e_weight_q,
        check_mxfp8_end_to_end,
    )
    torch.cuda.empty_cache()

    def warm_gpu(duration_seconds: float) -> None:
        if duration_seconds <= 0:
            return
        lhs = torch.empty((4096, 4096), device=device, dtype=torch.bfloat16).uniform_(
            -1.0, 1.0
        )
        rhs = torch.empty_like(lhs).uniform_(-1.0, 1.0)
        torch.cuda.synchronize()
        deadline = time.perf_counter() + duration_seconds
        while time.perf_counter() < deadline:
            for _ in range(10):
                torch.mm(lhs, rhs)
            torch.cuda.synchronize()
        del lhs, rhs
        torch.cuda.empty_cache()

    telemetry_fields = (
        ("timestamp", False),
        ("power.draw", True),
        ("power.limit", True),
        ("clocks.sm", True),
        ("clocks.max.sm", True),
        ("utilization.gpu", True),
        ("temperature.gpu", True),
        ("pstate", False),
        ("clocks_event_reasons.sw_power_cap", False),
        ("clocks_event_reasons.hw_thermal_slowdown", False),
    )
    telemetry_names = {
        "power.draw": "power_w",
        "power.limit": "power_limit_w",
        "clocks.sm": "sm_clock_mhz",
        "clocks.max.sm": "max_sm_clock_mhz",
        "utilization.gpu": "gpu_util_percent",
        "temperature.gpu": "temperature_c",
        "clocks_event_reasons.sw_power_cap": "sw_power_cap",
        "clocks_event_reasons.hw_thermal_slowdown": "hw_thermal_slowdown",
    }

    def parse_telemetry_value(value: str, numeric: bool):
        value = value.strip()
        if not numeric:
            return value
        try:
            return float(value)
        except ValueError:
            return None

    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=" + ",".join(field for field, _ in telemetry_fields),
            "--format=csv,noheader,nounits",
            f"--loop-ms={telemetry_interval_ms}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    monitor_started = time.perf_counter()
    telemetry_samples = []

    def read_telemetry() -> None:
        assert monitor.stdout is not None
        for line in monitor.stdout:
            values = next(csv.reader([line]))
            if len(values) != len(telemetry_fields):
                continue
            sample = {"host_time_s": time.perf_counter() - monitor_started}
            for (field, numeric), value in zip(telemetry_fields, values):
                sample[telemetry_names.get(field, field)] = parse_telemetry_value(
                    value, numeric
                )
            telemetry_samples.append(sample)

    monitor_reader = threading.Thread(target=read_telemetry, daemon=True)
    monitor_reader.start()

    def benchmark_block(fn) -> dict:
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        phase_start_s = time.perf_counter() - monitor_started
        wall_started = time.perf_counter()
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        wall_ms = 1000.0 * (time.perf_counter() - wall_started) / iterations
        phase_end_s = time.perf_counter() - monitor_started
        return {
            "gpu_latency_us": 1000.0 * float(start.elapsed_time(end)) / iterations,
            "wall_latency_us": 1000.0 * wall_ms,
            "phase_window_s": [phase_start_s, phase_end_s],
        }

    def summarize_samples(samples: list[dict], flop_count: int, roofline: float) -> dict:
        gpu_latencies = sorted(sample["gpu_latency_us"] for sample in samples)
        wall_latencies = sorted(sample["wall_latency_us"] for sample in samples)
        median_us = statistics.median(gpu_latencies)
        tflops = flop_count / median_us / 1e6
        return {
            "median_gpu_latency_us": median_us,
            "mean_gpu_latency_us": statistics.fmean(gpu_latencies),
            "min_gpu_latency_us": gpu_latencies[0],
            "max_gpu_latency_us": gpu_latencies[-1],
            "repeat_gpu_latency_us": gpu_latencies,
            "median_wall_latency_us": statistics.median(wall_latencies),
            "repeat_wall_latency_us": wall_latencies,
            "tflops": tflops,
            "roofline_tflops": roofline,
            "roofline_utilization_percent": 100.0 * tflops / roofline,
            "phase_windows_s": [sample["phase_window_s"] for sample in samples],
        }

    warm_gpu(gpu_warmup_seconds)
    points = []
    benchmark_started = time.perf_counter()
    for size in sizes:
        for input_mode in input_modes:
            print(f"benchmarking size={size} input_mode={input_mode}")
            torch.manual_seed(20260907 + size)
            torch.cuda.reset_peak_memory_stats()
            if input_mode == "random":
                x = torch.empty((size, size), device=device, dtype=torch.bfloat16)
                weight = torch.empty_like(x)
                x.uniform_(-1.0, 1.0)
                weight.uniform_(-1.0, 1.0)
            else:
                x = torch.ones((size, size), device=device, dtype=torch.bfloat16)
                weight = torch.ones_like(x)

            bf16_output = torch.empty_like(x)
            mxfp8_output = torch.empty_like(x)
            mxfp8_e2e_output = torch.empty_like(x)
            x_quantizer = new_quantizer()
            weight_quantizer = new_quantizer()
            x_q = quantize_rowwise(x_quantizer, x, "x")
            weight_q = quantize_rowwise(weight_quantizer, weight, "weight")

            def bf16_core() -> None:
                torch.mm(x, weight.T, out=bf16_output)

            def mxfp8_core() -> None:
                launch_mxfp8(weight_q, x_q, mxfp8_output)

            e2e_x_quantizer = new_quantizer()
            e2e_weight_quantizer = new_quantizer()

            def mxfp8_end_to_end() -> None:
                dynamic_x_q = e2e_x_quantizer.quantize(x)
                dynamic_weight_q = e2e_weight_quantizer.quantize(weight)
                launch_mxfp8(dynamic_weight_q, dynamic_x_q, mxfp8_e2e_output)

            methods = {
                "bf16_core": bf16_core,
                "mxfp8_core": mxfp8_core,
            }
            if include_end_to_end:
                methods["mxfp8_end_to_end"] = mxfp8_end_to_end

            for fn in methods.values():
                for _ in range(warmup):
                    fn()
                torch.cuda.synchronize()

            measured = {name: [] for name in methods}
            method_names = list(methods)
            for repeat in range(repeats):
                # Rotate the order so one method is not always measured in the
                # hottest or most power-limited position.
                offset = repeat % len(method_names)
                order = method_names[offset:] + method_names[:offset]
                for name in order:
                    measured[name].append(benchmark_block(methods[name]))

            flop_count = 2 * size**3
            method_results = {
                "bf16_core": summarize_samples(
                    measured["bf16_core"], flop_count, BF16_DENSE_ROOFLINE_TFLOPS
                ),
                "mxfp8_core": summarize_samples(
                    measured["mxfp8_core"], flop_count, MXFP8_DENSE_ROOFLINE_TFLOPS
                ),
            }
            if include_end_to_end:
                method_results["mxfp8_end_to_end"] = summarize_samples(
                    measured["mxfp8_end_to_end"],
                    flop_count,
                    MXFP8_DENSE_ROOFLINE_TFLOPS,
                )
            comparison = {
                "mxfp8_core_speedup_vs_bf16": (
                    method_results["bf16_core"]["median_gpu_latency_us"]
                    / method_results["mxfp8_core"]["median_gpu_latency_us"]
                ),
                "mxfp8_core_tflops_delta": (
                    method_results["mxfp8_core"]["tflops"]
                    - method_results["bf16_core"]["tflops"]
                ),
            }
            if include_end_to_end:
                comparison.update(
                    {
                        "mxfp8_end_to_end_speedup_vs_bf16": (
                            method_results["bf16_core"]["median_gpu_latency_us"]
                            / method_results["mxfp8_end_to_end"][
                                "median_gpu_latency_us"
                            ]
                        ),
                        "mxfp8_end_to_end_tflops_delta": (
                            method_results["mxfp8_end_to_end"]["tflops"]
                            - method_results["bf16_core"]["tflops"]
                        ),
                    }
                )
            points.append(
                {
                    "size": size,
                    "shape": {"m": size, "n": size, "k": size},
                    "input_mode": input_mode,
                    "flop_count": flop_count,
                    "methods": method_results,
                    "comparison": comparison,
                    "peak_allocated_memory_bytes": int(
                        torch.cuda.max_memory_allocated()
                    ),
                }
            )
            print(
                f"size={size} mode={input_mode} "
                f"BF16={method_results['bf16_core']['tflops']:.2f} TFLOPS "
                f"MXFP8={method_results['mxfp8_core']['tflops']:.2f} TFLOPS "
                f"speedup={comparison['mxfp8_core_speedup_vs_bf16']:.3f}x"
            )
            del (
                x,
                weight,
                bf16_output,
                mxfp8_output,
                mxfp8_e2e_output,
                x_q,
                weight_q,
            )
            torch.cuda.empty_cache()

    benchmark_seconds = time.perf_counter() - benchmark_started
    time.sleep(0.1)
    monitor.terminate()
    try:
        monitor.wait(timeout=5)
    except subprocess.TimeoutExpired:
        monitor.kill()
        monitor.wait(timeout=5)
    monitor_reader.join(timeout=5)
    monitor_stderr = (
        monitor.stderr.read().strip() if monitor.stderr is not None else ""
    )

    def numeric_stats(samples: list[dict], key: str):
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

    def telemetry_for(windows: list[list[float]]) -> dict:
        selected = [
            sample
            for sample in telemetry_samples
            if any(start <= sample["host_time_s"] <= end for start, end in windows)
        ]
        power_cap_count = sum(
            sample.get("sw_power_cap") == "Active" for sample in selected
        )
        return {
            "sample_count": len(selected),
            "power_w": numeric_stats(selected, "power_w"),
            "sm_clock_mhz": numeric_stats(selected, "sm_clock_mhz"),
            "gpu_util_percent": numeric_stats(selected, "gpu_util_percent"),
            "temperature_c": numeric_stats(selected, "temperature_c"),
            "sw_power_cap_active_count": power_cap_count,
            "sw_power_cap_active_percent": (
                100.0 * power_cap_count / len(selected) if selected else None
            ),
            "hw_thermal_slowdown_values": sorted(
                {
                    sample.get("hw_thermal_slowdown")
                    for sample in selected
                    if sample.get("hw_thermal_slowdown") is not None
                }
            ),
        }

    for point in points:
        for method in point["methods"].values():
            method["telemetry"] = telemetry_for(method.pop("phase_windows_s"))

    def peak_for(input_mode: str, method_name: str):
        candidates = [
            point
            for point in points
            if point["input_mode"] == input_mode and method_name in point["methods"]
        ]
        if not candidates:
            return None
        best = max(candidates, key=lambda point: point["methods"][method_name]["tflops"])
        return {
            "size": best["size"],
            "tflops": best["methods"][method_name]["tflops"],
            "median_gpu_latency_us": best["methods"][method_name][
                "median_gpu_latency_us"
            ],
            "roofline_utilization_percent": best["methods"][method_name][
                "roofline_utilization_percent"
            ],
        }

    summary = {}
    for input_mode in input_modes:
        mode_summary = {
            "bf16_core_peak": peak_for(input_mode, "bf16_core"),
            "mxfp8_core_peak": peak_for(input_mode, "mxfp8_core"),
        }
        if include_end_to_end:
            mode_summary["mxfp8_end_to_end_peak"] = peak_for(
                input_mode, "mxfp8_end_to_end"
            )
        bf16_peak = mode_summary["bf16_core_peak"]
        mxfp8_peak = mode_summary["mxfp8_core_peak"]
        mode_summary["independent_peak_tflops_ratio"] = (
            mxfp8_peak["tflops"] / bf16_peak["tflops"]
        )
        mode_summary["independent_peak_tflops_delta"] = (
            mxfp8_peak["tflops"] - bf16_peak["tflops"]
        )
        summary[input_mode] = mode_summary

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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    remote_run_dir = (
        Path(CACHE_MOUNT)
        / "runs"
        / _safe_slug(str(properties.name))
        / run_id
        / KERNEL_NAME
    )
    remote_run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS",
        "run_id": run_id,
        "kernel": KERNEL_NAME,
        "operation": "Y[M,N] = X[M,K] @ W[N,K].T",
        "precision": {
            "api_inputs": "BF16 X and BF16 W for both paths",
            "output": "BF16 for both paths",
            "bf16_core": "BF16 multiply; torch.mm Tensor Core path",
            "mxfp8_core": (
                "rowwise E4M3 values with one E8M0 scale per 32 K values; "
                "BF16 -> MXFP8 quantization excluded from timing"
            ),
            "mxfp8_end_to_end": (
                "BF16 -> rowwise MXFP8 quantization of X and W included in timing; "
                "effective GEMM TFLOPS"
            ),
            "mxfp8_compute_contract": "CUBLAS_COMPUTE_32F via Transformer Engine",
            "mxfp8_fast_accum_requested": fast_accum,
            "mxfp8_use_split_accumulator": not fast_accum,
        },
        "correctness": correctness,
        "benchmark": {
            "sizes": list(sizes),
            "input_modes": list(input_modes),
            "timer": "CUDA events around batches of launches",
            "gpu_warmup_seconds": gpu_warmup_seconds,
            "warmup_per_method_and_point": warmup,
            "iterations_per_repeat": iterations,
            "repeats": repeats,
            "statistic": "median repeat-average GPU latency",
            "rotated_method_order": True,
            "benchmark_seconds": benchmark_seconds,
        },
        "rooflines": {
            "bf16_dense_tflops": BF16_DENSE_ROOFLINE_TFLOPS,
            "mxfp8_dense_tflops": MXFP8_DENSE_ROOFLINE_TFLOPS,
            "nominal_ratio": (
                MXFP8_DENSE_ROOFLINE_TFLOPS / BF16_DENSE_ROOFLINE_TFLOPS
            ),
            "source": ROOFLINE_SOURCE,
        },
        "summary": summary,
        "points": points,
        "telemetry": {
            "requested_interval_ms": telemetry_interval_ms,
            "monitor_stderr": monitor_stderr,
            "samples": telemetry_samples,
        },
        "environment": {
            "gpu_name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "sm_count": int(properties.multi_processor_count),
            "gpu_memory_bytes": int(properties.total_memory),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "transformer_engine": str(transformer_engine.__version__),
            "mxfp8_available": bool(mxfp8_available),
            "mxfp8_reason": str(mxfp8_reason),
            "nvidia_smi": static_smi,
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
    sizes: str = ",".join(str(size) for size in DEFAULT_SIZES),
    input_modes: str = ",".join(DEFAULT_INPUT_MODES),
    warmup: int = 20,
    iterations: int = 20,
    repeats: int = 5,
    gpu_warmup_seconds: float = 10.0,
    telemetry_interval_ms: int = 20,
    include_end_to_end: bool = True,
    fast_accum: bool = True,
):
    parsed_sizes = _parse_csv_ints(sizes)
    parsed_modes = _parse_csv_strings(input_modes)
    result = run_remote.remote(
        sizes=parsed_sizes,
        input_modes=parsed_modes,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        gpu_warmup_seconds=gpu_warmup_seconds,
        telemetry_interval_ms=telemetry_interval_ms,
        include_end_to_end=include_end_to_end,
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
                "summary": result["summary"],
                "correctness": result["correctness"],
                "environment": result["environment"],
                "local_artifact": str(result_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
