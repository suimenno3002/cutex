"""Benchmark direct cuBLAS BF16 GEMM with FP32 accumulation on Modal B300."""

from __future__ import annotations

import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-cublas-bf16-fp32acc-b300"
CACHE_MOUNT = "/cutex-cache"
KERNEL_BASE_NAME = "cublas_bf16_fp32acc_gemm"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
LOCAL_CUDA_SOURCE = Path(__file__).with_name("cublas_bf16_gemm.cu")
REMOTE_CUDA_SOURCE = "/opt/cutex/cublas_bf16_gemm.cu"
REMOTE_BINARY = "/tmp/cublas_bf16_gemm"
ROOFLINE_TFLOPS = 2250.0
PHASE_ORDER = ("random", "one", "one", "random")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("cutex-autotune-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.07-py3")
    .entrypoint([])
    .env({"NVIDIA_IMEX_CHANNELS": "0", "PYTHONUNBUFFERED": "1"})
    .add_local_file(str(LOCAL_CUDA_SOURCE), REMOTE_CUDA_SOURCE)
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
    warmup: int = 100,
    iterations: int = 200,
    repeats: int = 5,
    sample_interval_ms: int = 20,
    idle_seconds: float = 5.0,
    input_mode: str = "both",
    launch_interval_ms: int = 0,
    per_sample_warmup: int = 0,
    per_sample_warmup_mode: str = "same",
    protocol_sweep: bool = False,
) -> dict:
    import statistics
    import subprocess
    import threading
    import time

    import torch

    if size <= 0 or size % 8:
        raise ValueError("size must be a positive multiple of 8")
    if warmup < 0 or iterations < 1 or repeats < 1:
        raise ValueError("invalid benchmark iteration setting")
    if (
        sample_interval_ms < 5
        or idle_seconds < 0
        or launch_interval_ms < 0
        or per_sample_warmup < 0
    ):
        raise ValueError("invalid telemetry setting")
    if input_mode not in ("both", "random", "one"):
        raise ValueError("input_mode must be 'both', 'random', or 'one'")
    if per_sample_warmup_mode not in ("same", "one"):
        raise ValueError("per_sample_warmup_mode must be 'same' or 'one'")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal did not attach a CUDA-capable GPU")
    properties = torch.cuda.get_device_properties(0)
    if (properties.major, properties.minor) != (10, 3):
        raise RuntimeError(f"expected B300/SM103, got {properties.name}")
    phase_order = PHASE_ORDER if input_mode == "both" else (input_mode,)
    kernel_name = f"{KERNEL_BASE_NAME}_{size}"
    if protocol_sweep:
        kernel_name += f"_protocol_sweep_warmup-{per_sample_warmup_mode}"
        phase_specs = [
            {
                "mode": "random",
                "interval_ms": interval,
                "sample_warmup": count,
                "sample_warmup_mode": per_sample_warmup_mode,
            }
            for count in (0, 1, 2, 4, 8, 16, 32, 64)
            for interval in (20, 50, 100, 250, 500, 1000)
        ]
    else:
        phase_specs = [
            {
                "mode": mode,
                "interval_ms": launch_interval_ms,
                "sample_warmup": per_sample_warmup,
                "sample_warmup_mode": per_sample_warmup_mode,
            }
            for mode in phase_order
        ]
    if launch_interval_ms and not protocol_sweep:
        kernel_name += f"_spaced_{launch_interval_ms}ms_{input_mode}"
        if per_sample_warmup:
            kernel_name += f"_warmup{per_sample_warmup}-{per_sample_warmup_mode}"

    compile_result = subprocess.run(
        [
            "nvcc",
            "-O3",
            "-std=c++17",
            "-arch=sm_103",
            REMOTE_CUDA_SOURCE,
            "-lcublas",
            "-o",
            REMOTE_BINARY,
        ],
        capture_output=True,
        text=True,
    )
    if compile_result.returncode:
        raise RuntimeError(f"nvcc failed:\n{compile_result.stderr}")

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
    friendly_names = {
        "power.draw": "power_w",
        "power.limit": "power_limit_w",
        "clocks.sm": "sm_clock_mhz",
        "clocks.max.sm": "max_sm_clock_mhz",
        "utilization.gpu": "gpu_util_percent",
        "temperature.gpu": "temperature_c",
        "clocks_event_reasons.sw_power_cap": "sw_power_cap",
        "clocks_event_reasons.hw_thermal_slowdown": "hw_thermal_slowdown",
    }
    smi_query = ",".join(field for field, _numeric in telemetry_fields)

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
        started = time.perf_counter()
        samples = []

        def read_samples():
            assert process.stdout is not None
            for line in process.stdout:
                values = next(csv.reader([line]))
                if len(values) != len(telemetry_fields):
                    continue
                sample = {"host_time_s": time.perf_counter() - started}
                for (field, numeric), value in zip(telemetry_fields, values):
                    sample[friendly_names.get(field, field)] = parse_value(value, numeric)
                samples.append(sample)

        reader = threading.Thread(target=read_samples, daemon=True)
        reader.start()
        return process, reader, started, samples

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

    phases = []
    for phase_index, phase_spec in enumerate(phase_specs):
        mode = phase_spec["mode"]
        phase_interval_ms = phase_spec["interval_ms"]
        phase_sample_warmup = phase_spec["sample_warmup"]
        phase_sample_warmup_mode = phase_spec["sample_warmup_mode"]
        monitor, reader, monitor_started, samples = start_monitor()
        time.sleep(idle_seconds)
        benchmark_start_s = None
        benchmark_end_s = None
        payload = None
        process = subprocess.Popen(
            [
                REMOTE_BINARY,
                "--n",
                str(size),
                "--warmup",
                str(warmup),
                "--iterations",
                str(iterations),
                "--repeats",
                str(repeats),
                "--interval-ms",
                str(phase_interval_ms),
                "--sample-warmup",
                str(phase_sample_warmup),
                "--sample-warmup-mode",
                phase_sample_warmup_mode,
                "--mode",
                mode,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            line = line.strip()
            if line == "CUTEX_BENCH_START":
                benchmark_start_s = time.perf_counter() - monitor_started
            elif line == "CUTEX_BENCH_END":
                benchmark_end_s = time.perf_counter() - monitor_started
            elif line.startswith("{"):
                payload = json.loads(line)
        process.wait(timeout=30 * 60)
        stderr = process.stderr.read().strip() if process.stderr is not None else ""
        if process.returncode or payload is None:
            raise RuntimeError(f"cuBLAS benchmark failed ({process.returncode}): {stderr}")
        if benchmark_start_s is None or benchmark_end_s is None:
            raise RuntimeError("cuBLAS benchmark markers were not received")
        time.sleep(0.5)
        monitor_stderr = stop_monitor(monitor, reader)
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
        telemetry = {
            "sample_count": len(benchmark_samples),
            "power_w": numeric_stats(benchmark_samples, "power_w"),
            "sm_clock_mhz": numeric_stats(benchmark_samples, "sm_clock_mhz"),
            "gpu_util_percent": numeric_stats(benchmark_samples, "gpu_util_percent"),
            "temperature_c": numeric_stats(benchmark_samples, "temperature_c"),
            "sw_power_cap_active_count": active_count,
            "sw_power_cap_active_percent": 100.0 * active_count / len(benchmark_samples),
            "hw_thermal_slowdown_values": sorted(
                {sample.get("hw_thermal_slowdown") for sample in benchmark_samples}
            ),
        }
        payload["roofline_tflops"] = ROOFLINE_TFLOPS
        payload["roofline_utilization_percent"] = 100.0 * payload["tflops"] / ROOFLINE_TFLOPS
        phases.append(
            {
                "phase_index": phase_index,
                "mode": mode,
                "protocol": {
                    "launch_interval_ms": phase_interval_ms,
                    "per_sample_warmup": phase_sample_warmup,
                    "per_sample_warmup_mode": phase_sample_warmup_mode,
                },
                "benchmark": payload,
                "telemetry": telemetry,
                "benchmark_start_s": benchmark_start_s,
                "benchmark_end_s": benchmark_end_s,
                "monitor_stderr": monitor_stderr,
                "samples": samples,
            }
        )
        print(
            f"phase={phase_index} mode={mode} tflops={payload['tflops']:.2f} "
            f"power={telemetry['power_w']['mean']:.1f}W "
            f"clock={telemetry['sm_clock_mhz']['mean']:.1f}MHz"
        )

    if protocol_sweep:
        candidates = [
            {
                **phase["protocol"],
                "phase_index": phase["phase_index"],
                "latency_us": 1000.0 * phase["benchmark"]["median_ms"],
                "tflops": phase["benchmark"]["tflops"],
                "sw_power_cap_active_percent": phase["telemetry"][
                    "sw_power_cap_active_percent"
                ],
                "telemetry_sample_count": phase["telemetry"]["sample_count"],
            }
            for phase in phases
        ]
        eligible = [
            candidate
            for candidate in candidates
            if candidate["sw_power_cap_active_percent"] == 0.0
        ]
        summaries = {
            "candidates": sorted(candidates, key=lambda item: item["tflops"], reverse=True),
            "best_without_observed_power_cap": (
                max(eligible, key=lambda item: item["tflops"]) if eligible else None
            ),
        }
    else:
        summaries = {}
        for mode in dict.fromkeys(phase_order):
            selected = [phase for phase in phases if phase["mode"] == mode]
            summaries[mode] = {
                "phase_indices": [phase["phase_index"] for phase in selected],
                "tflops": [phase["benchmark"]["tflops"] for phase in selected],
                "tflops_mean": statistics.fmean(
                    phase["benchmark"]["tflops"] for phase in selected
                ),
                "latency_ms": [phase["benchmark"]["median_ms"] for phase in selected],
                "latency_ms_mean": statistics.fmean(
                    phase["benchmark"]["median_ms"] for phase in selected
                ),
                "power_w_mean": [
                    phase["telemetry"]["power_w"]["mean"] for phase in selected
                ],
                "sm_clock_mhz_mean": [
                    phase["telemetry"]["sm_clock_mhz"]["mean"] for phase in selected
                ],
                "sw_power_cap_active_percent": [
                    phase["telemetry"]["sw_power_cap_active_percent"]
                    for phase in selected
                ],
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
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    remote_run_dir = (
        Path(CACHE_MOUNT)
        / "runs"
        / _safe_slug(str(properties.name))
        / run_id
        / kernel_name
    )
    remote_run_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS",
        "run_id": run_id,
        "kernel": kernel_name,
        "operation": "column-major C[N,N] = A[N,N] @ B[N,N]",
        "shape": {"m": size, "n": size, "k": size},
        "precision": {
            "a": "CUDA_R_16BF",
            "b": "CUDA_R_16BF",
            "c": "CUDA_R_16BF",
            "compute_and_accumulation": "CUBLAS_COMPUTE_32F",
            "api": "cublasGemmEx",
            "algorithm": "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
        },
        "methodology": {
            "phase_order": [phase["mode"] for phase in phase_specs],
            "input_mode": input_mode,
            "launch_interval_ms": launch_interval_ms,
            "per_sample_warmup": per_sample_warmup,
            "per_sample_warmup_mode": per_sample_warmup_mode,
            "protocol_sweep": protocol_sweep,
            "warmup_per_phase": warmup,
            "iterations_per_repeat": iterations,
            "repeats_per_phase": repeats,
            "statistic": "median of repeat-average CUDA-event latency",
            "sample_interval_ms_requested": sample_interval_ms,
            "idle_seconds_before_phase": idle_seconds,
            "telemetry_scope": (
                "wall interval including launch spacing"
                if launch_interval_ms or protocol_sweep
                else "continuous measured repeat window"
            ),
            "roofline_tflops": ROOFLINE_TFLOPS,
        },
        "summary": summaries,
        "phases": phases,
        "environment": {
            "gpu_name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "gpu_memory_bytes": int(properties.total_memory),
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "nvidia_smi": static_smi,
            "nvcc_stderr": compile_result.stderr.strip(),
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
    warmup: int = 100,
    iterations: int = 200,
    repeats: int = 5,
    sample_interval_ms: int = 20,
    idle_seconds: float = 5.0,
    input_mode: str = "both",
    launch_interval_ms: int = 0,
    per_sample_warmup: int = 0,
    per_sample_warmup_mode: str = "same",
    protocol_sweep: bool = False,
):
    result = run_remote.remote(
        size=size,
        warmup=warmup,
        iterations=iterations,
        repeats=repeats,
        sample_interval_ms=sample_interval_ms,
        idle_seconds=idle_seconds,
        input_mode=input_mode,
        launch_interval_ms=launch_interval_ms,
        per_sample_warmup=per_sample_warmup,
        per_sample_warmup_mode=per_sample_warmup_mode,
        protocol_sweep=protocol_sweep,
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
                "methodology": result["methodology"],
                "summary": result["summary"],
                "environment": result["environment"],
                "local_artifact": str(result_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
