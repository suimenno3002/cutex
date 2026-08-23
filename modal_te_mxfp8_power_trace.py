"""Correlate B300 MXFP8 GEMM throughput with power and SM clocks."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-te-mxfp8-power-trace-b300"
CACHE_MOUNT = "/cutex-cache"
KERNEL_NAME = "te_mxfp8_power_trace_16384"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
ROOFLINE_TFLOPS = 4500.0
BATCH_COUNTS = (1, 8, 32, 128, 512, 2048)
PHASE_ORDER = ("random", "zero", "zero", "random")
EXTENDED_PHASE_ORDER = (
    "random",
    "zero",
    "one",
    "random_sign",
    "random_sign",
    "one",
    "zero",
    "random",
)

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
    size: int = 16384,
    sample_interval_ms: int = 20,
    idle_seconds: float = 5.0,
    extended_patterns: bool = False,
) -> dict:
    import csv
    import math
    import statistics
    import subprocess
    import threading
    import time

    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    if size <= 0 or size % 32:
        raise ValueError("size must be a positive multiple of 32")
    if sample_interval_ms < 5 or idle_seconds < 0:
        raise ValueError("sample_interval_ms must be >= 5 and idle_seconds >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")
    availability = te.is_mxfp8_available(return_reason=True)
    available, reason = (
        availability if isinstance(availability, tuple) else (bool(availability), "")
    )
    if not available:
        raise RuntimeError(f"Transformer Engine MXFP8 is unavailable: {reason}")

    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (10, 3):
        raise RuntimeError(
            f"this experiment targets B300/SM103; got {properties.name} "
            f"(SM{properties.major}{properties.minor})"
        )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    kernel_name = KERNEL_NAME + ("_patterns" if extended_patterns else "")
    phase_order = EXTENDED_PHASE_ORDER if extended_patterns else PHASE_ORDER
    remote_run_dir = (
        Path(CACHE_MOUNT)
        / "runs"
        / _safe_slug(str(properties.name))
        / run_id
        / kernel_name
    )
    remote_run_dir.mkdir(parents=True, exist_ok=True)

    candidates = (
        ("timestamp", "timestamp", False),
        ("power_w", "power.draw", True),
        ("power_limit_w", "power.limit", True),
        ("sm_clock_mhz", "clocks.sm", True),
        ("max_sm_clock_mhz", "clocks.max.sm", True),
        ("gpu_util_percent", "utilization.gpu", True),
        ("memory_util_percent", "utilization.memory", True),
        ("temperature_c", "temperature.gpu", True),
        ("pstate", "pstate", False),
        ("sw_power_cap", "clocks_event_reasons.sw_power_cap", False),
        ("hw_slowdown", "clocks_event_reasons.hw_slowdown", False),
        (
            "hw_thermal_slowdown",
            "clocks_event_reasons.hw_thermal_slowdown",
            False,
        ),
        (
            "hw_power_brake_slowdown",
            "clocks_event_reasons.hw_power_brake_slowdown",
            False,
        ),
    )

    def supported_smi_fields():
        supported = []
        for friendly, query, numeric in candidates:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                supported.append((friendly, query, numeric))
        if not {entry[0] for entry in supported} >= {"power_w", "sm_clock_mhz"}:
            raise RuntimeError(f"required nvidia-smi fields unavailable: {supported}")
        return supported

    smi_fields = supported_smi_fields()
    smi_query = ",".join(entry[1] for entry in smi_fields)

    def parse_value(value: str, numeric: bool):
        value = value.strip()
        if not numeric:
            return value
        try:
            return float(value)
        except ValueError:
            return None

    def start_monitor():
        command = [
            "nvidia-smi",
            f"--query-gpu={smi_query}",
            "--format=csv,noheader,nounits",
            f"--loop-ms={sample_interval_ms}",
        ]
        process = subprocess.Popen(
            command,
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
                if len(values) != len(smi_fields):
                    continue
                sample = {"host_time_s": time.perf_counter() - monitor_started}
                for (friendly, _query, numeric), value in zip(smi_fields, values):
                    sample[friendly] = parse_value(value, numeric)
                samples.append(sample)

        reader = threading.Thread(target=read_samples, daemon=True)
        reader.start()
        return {
            "command": command,
            "process": process,
            "reader": reader,
            "started": monitor_started,
            "samples": samples,
        }

    def stop_monitor(monitor):
        process = monitor["process"]
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        monitor["reader"].join(timeout=5)
        stderr = process.stderr.read().strip() if process.stderr is not None else ""
        if len(monitor["samples"]) < 2:
            raise RuntimeError(
                f"nvidia-smi monitoring produced too few samples: {stderr!r}"
            )
        return stderr

    def numeric_stats(samples, key: str):
        values = [sample[key] for sample in samples if sample.get(key) is not None]
        if not values:
            return None
        ordered = sorted(values)
        return {
            "count": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "min": ordered[0],
            "max": ordered[-1],
            "p95": ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)],
        }

    def summarize_samples(samples):
        return {
            key: numeric_stats(samples, key)
            for key in (
                "power_w",
                "sm_clock_mhz",
                "gpu_util_percent",
                "memory_util_percent",
                "temperature_c",
            )
        } | {
            key: sorted(
                {sample[key] for sample in samples if sample.get(key) not in (None, "")}
            )
            for key in (
                "pstate",
                "sw_power_cap",
                "hw_slowdown",
                "hw_thermal_slowdown",
                "hw_power_brake_slowdown",
            )
            if any(key in sample for sample in samples)
        }

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

    torch.manual_seed(20260822)
    dtype = torch.bfloat16
    def make_inputs(mode: str):
        if mode == "random":
            x = torch.randn((size, size), device="cuda", dtype=dtype)
            weight = torch.randn_like(x)
        elif mode == "zero":
            x = torch.zeros((size, size), device="cuda", dtype=dtype)
            weight = torch.zeros_like(x)
        elif mode == "one":
            x = torch.ones((size, size), device="cuda", dtype=dtype)
            weight = torch.ones_like(x)
        elif mode == "random_sign":
            x = torch.empty((size, size), device="cuda", dtype=dtype)
            weight = torch.empty_like(x)
            x.bernoulli_(0.5).mul_(2).sub_(1)
            weight.bernoulli_(0.5).mul_(2).sub_(1)
        else:
            raise ValueError(f"unknown input mode: {mode}")
        quantized = (quantize_rowwise(x), quantize_rowwise(weight))
        del x, weight
        return quantized

    modes = tuple(dict.fromkeys(phase_order))
    inputs = {mode: make_inputs(mode) for mode in modes}
    output = torch.empty((size, size), device="cuda", dtype=dtype)

    def fprop(mode: str):
        x_q, weight_q = inputs[mode]
        general_gemm(
            weight_q,
            x_q,
            out_dtype=dtype,
            out=output,
            layout="TN",
            use_split_accumulator=True,
        )

    # Compile both paths before collecting cold-to-sustained phase data.
    validation = {}
    for mode in modes:
        fprop(mode)
        torch.cuda.synchronize()
        sampled = output[::256, ::256].float()
        validation[mode] = {
            "all_finite": bool(torch.isfinite(output).all().item()),
            "sample_mean_abs": float(sampled.abs().mean().item()),
            "sample_max_abs": float(sampled.abs().max().item()),
        }
        if not validation[mode]["all_finite"]:
            raise AssertionError(f"{mode} output contains non-finite values")
        if mode == "zero" and validation[mode]["sample_max_abs"] != 0.0:
            raise AssertionError(f"zero-input output is not zero: {validation[mode]}")
        if mode == "one" and validation[mode]["sample_mean_abs"] != float(size):
            raise AssertionError(f"one-input output is unexpected: {validation[mode]}")

    flop_count = 2 * size**3
    phases = []
    for phase_index, mode in enumerate(phase_order):
        monitor = start_monitor()
        time.sleep(idle_seconds)
        phase_start_s = time.perf_counter() - monitor["started"]
        segments = []
        for launches in BATCH_COUNTS:
            host_start_s = time.perf_counter() - monitor["started"]
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(launches):
                fprop(mode)
            end_event.record()
            end_event.synchronize()
            host_end_s = time.perf_counter() - monitor["started"]
            elapsed_ms = float(start_event.elapsed_time(end_event))
            latency_us = elapsed_ms * 1000.0 / launches
            segments.append(
                {
                    "launches": launches,
                    "elapsed_ms": elapsed_ms,
                    "latency_us": latency_us,
                    "tflops": flop_count / latency_us / 1e6,
                    "roofline_utilization_percent": (
                        100.0 * flop_count / latency_us / 1e6 / ROOFLINE_TFLOPS
                    ),
                    "host_start_s": host_start_s,
                    "host_end_s": host_end_s,
                }
            )
        phase_end_s = time.perf_counter() - monitor["started"]
        time.sleep(0.5)
        monitor_stderr = stop_monitor(monitor)
        phase_samples = [
            sample
            for sample in monitor["samples"]
            if phase_start_s <= sample["host_time_s"] <= phase_end_s
        ]
        for segment in segments:
            segment_samples = [
                sample
                for sample in phase_samples
                if segment["host_start_s"]
                <= sample["host_time_s"]
                <= segment["host_end_s"]
            ]
            segment["telemetry"] = summarize_samples(segment_samples)
            segment["telemetry_sample_count"] = len(segment_samples)
        total_elapsed_ms = sum(segment["elapsed_ms"] for segment in segments)
        total_launches = sum(segment["launches"] for segment in segments)
        phases.append(
            {
                "phase_index": phase_index,
                "mode": mode,
                "idle_seconds_before_phase": idle_seconds,
                "phase_start_s": phase_start_s,
                "phase_end_s": phase_end_s,
                "total_launches": total_launches,
                "total_elapsed_ms": total_elapsed_ms,
                "weighted_latency_us": total_elapsed_ms * 1000.0 / total_launches,
                "weighted_tflops": (
                    flop_count * total_launches / total_elapsed_ms / 1e9
                ),
                "telemetry": summarize_samples(phase_samples),
                "telemetry_sample_count": len(phase_samples),
                "segments": segments,
                "monitor_stderr": monitor_stderr,
                "samples": monitor["samples"],
            }
        )
        print(
            f"phase={phase_index} mode={mode} "
            f"weighted_tflops={phases[-1]['weighted_tflops']:.2f} "
            f"samples={len(phase_samples)}"
        )

    def mode_summary(mode: str):
        selected = [phase for phase in phases if phase["mode"] == mode]
        sustained = [
            next(
                segment
                for segment in phase["segments"]
                if segment["launches"] == BATCH_COUNTS[-1]
            )
            for phase in selected
        ]
        return {
            "phase_indices": [phase["phase_index"] for phase in selected],
            "weighted_tflops": [phase["weighted_tflops"] for phase in selected],
            "sustained_launches": BATCH_COUNTS[-1],
            "sustained_tflops": [segment["tflops"] for segment in sustained],
            "sustained_tflops_mean": statistics.fmean(
                segment["tflops"] for segment in sustained
            ),
            "sustained_power_w_mean": [
                (
                    segment["telemetry"]["power_w"]["mean"]
                    if segment["telemetry"]["power_w"]
                    else None
                )
                for segment in sustained
            ],
            "sustained_sm_clock_mhz_mean": [
                (
                    segment["telemetry"]["sm_clock_mhz"]["mean"]
                    if segment["telemetry"]["sm_clock_mhz"]
                    else None
                )
                for segment in sustained
            ],
        }

    summaries = {mode: mode_summary(mode) for mode in modes}
    summaries["relative_to_random_sustained_tflops_percent"] = {
        mode: 100.0
        * (
            summaries[mode]["sustained_tflops_mean"]
            / summaries["random"]["sustained_tflops_mean"]
            - 1.0
        )
        for mode in modes
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
        "operation": "Y[M,N] = X[M,K] @ W[N,K].T",
        "shape": {"m": size, "n": size, "k": size},
        "precision": {
            "inputs": "rowwise-only MXFP8 E4M3, one E8M0 scale per 32 K values",
            "multiply": "FP8 E4M3 x E4M3",
            "compute_type": "CUBLAS_COMPUTE_32F",
            "accumulation": "FP32 compute contract; split accumulator enabled",
            "output": "BF16",
            "quantization_in_timed_region": False,
        },
        "experiment": {
            "purpose": "separate sustained power/DVFS effects from input-independent kernel overhead",
            "phase_order": list(phase_order),
            "extended_patterns": extended_patterns,
            "batch_counts": list(BATCH_COUNTS),
            "sample_interval_ms_requested": sample_interval_ms,
            "idle_seconds_before_each_phase": idle_seconds,
            "timer": "CUDA events",
            "telemetry": "nvidia-smi loop sampled by a host reader thread",
            "roofline_tflops": ROOFLINE_TFLOPS,
        },
        "validation": validation,
        "summary": summaries,
        "phases": phases,
        "environment": {
            "gpu_name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": int(properties.total_memory),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "transformer_engine": str(transformer_engine.__version__),
            "mxfp8_available": bool(available),
            "mxfp8_reason": str(reason),
            "nvidia_smi": static_smi,
            "telemetry_fields": [
                {"name": name, "query": query, "numeric": numeric}
                for name, query, numeric in smi_fields
            ],
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
    size: int = 16384,
    sample_interval_ms: int = 20,
    idle_seconds: float = 5.0,
    extended_patterns: bool = False,
):
    result = run_remote.remote(
        size=size,
        sample_interval_ms=sample_interval_ms,
        idle_seconds=idle_seconds,
        extended_patterns=extended_patterns,
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
                "shape": result["shape"],
                "precision": result["precision"],
                "experiment": result["experiment"],
                "validation": result["validation"],
                "summary": result["summary"],
                "environment": result["environment"],
                "local_artifact": str(result_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
