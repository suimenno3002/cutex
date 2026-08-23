"""Profile the B300 rowwise MXFP8 Fprop kernel with Nsight Compute."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import modal


app = modal.App("cutex-te-mxfp8-ncu")
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.07-py3")
    .entrypoint([])
    .env({"NVIDIA_IMEX_CHANNELS": "0", "PYTHONUNBUFFERED": "1"})
)


@app.function(image=image, gpu="B300", timeout=15 * 60)
def profile_remote() -> dict:
    import shutil
    import subprocess
    import sys

    ncu = shutil.which("ncu")
    if ncu is None:
        return {"status": "NCU_NOT_FOUND"}

    version = subprocess.run(
        [ncu, "--version"], capture_output=True, text=True, check=False
    )
    query = subprocess.run(
        [ncu, "--query-metrics"], capture_output=True, text=True, check=False
    )
    query_text = query.stdout + "\n" + query.stderr
    metric_bases = set()
    tensor_metric_lines = []
    for line in query_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        base = stripped.split()[0].rstrip(",")
        if base.startswith(("sm__", "smsp__", "gpu__", "dram__")):
            metric_bases.add(base)
        if "tensor" in stripped.lower():
            tensor_metric_lines.append(stripped)

    requested = []

    def add(base: str, suffix: str) -> None:
        if base in metric_bases:
            requested.append(base + suffix)

    for base in (
        "sm__pipe_tensor_cycles_active",
        "sm__pipe_tensor_op_hmma_cycles_active",
        "sm__pipe_tensor_op_mma_cycles_active",
    ):
        add(base, ".avg.pct_of_peak_sustained_elapsed")
        add(base, ".avg.pct_of_peak_sustained_active")
    add("sm__throughput", ".avg.pct_of_peak_sustained_elapsed")
    add("dram__throughput", ".avg.pct_of_peak_sustained_elapsed")
    add("sm__cycles_elapsed", ".avg.per_second")
    add("gpu__time_duration", ".sum")

    if not requested:
        requested = [
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__cycles_elapsed.avg.per_second",
            "gpu__time_duration.sum",
        ]

    worker = r'''
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions import general_gemm

torch.manual_seed(20260822)
device = torch.device("cuda")
dtype = torch.bfloat16
size = 4096
x = torch.randn((size, size), device=device, dtype=dtype)
weight = torch.randn((size, size), device=device, dtype=dtype)

def quantize(value):
    q = te.MXFP8Quantizer(
        tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    q.optimize_for_gemm = True
    return q.quantize(value)

x_q = quantize(x)
weight_q = quantize(weight)
out = torch.empty((size, size), device=device, dtype=dtype)

def fprop():
    general_gemm(
        weight_q,
        x_q,
        out_dtype=dtype,
        out=out,
        layout="TN",
        use_split_accumulator=True,
    )

for _ in range(200):
    fprop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
fprop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
print("WORKER_DONE", float(out[0, 0]))
'''
    command = [
        ncu,
        "--profile-from-start",
        "off",
        "--target-processes",
        "all",
        "--clock-control",
        "none",
        "--replay-mode",
        "kernel",
        "--launch-count",
        "1",
        "--page",
        "raw",
        "--csv",
        "--metrics",
        ",".join(requested),
        sys.executable,
        "-c",
        worker,
    ]
    profile = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "status": "PASS" if profile.returncode == 0 else "PROFILE_FAILED",
        "ncu_path": ncu,
        "ncu_version": (version.stdout + version.stderr).strip(),
        "query_returncode": query.returncode,
        "query_output_tail": query_text[-20_000:],
        "tensor_metric_lines": tensor_metric_lines[:200],
        "requested_metrics": requested,
        "profile_returncode": profile.returncode,
        "profile_stdout": profile.stdout[-100_000:],
        "profile_stderr": profile.stderr[-100_000:],
    }


@app.local_entrypoint()
def main():
    result = profile_remote.remote()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir = Path(__file__).parent / "artifacts" / f"{run_id}-te_mxfp8_ncu"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({**result, "local_artifact": str(result_path.resolve())}, indent=2))
