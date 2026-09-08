"""Compile, verify, benchmark, and optionally trace the fixed MXFP8 GEMM on B300."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-dense-gemm-mxfp8-b300"
CACHE_MOUNT = "/cutex-cache"
KERNEL_BASE_NAME = "dense_gemm_mxfp8_16384"
LOCAL_ARTIFACT_ROOT = Path(__file__).parent / "artifacts"

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name("cutex-autotune-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.07-py3")
    .entrypoint([])
    .uv_pip_install("nvidia-cutlass-dsl[cu13]==4.7.0")
    .env({"NVIDIA_IMEX_CHANNELS": "0", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("cutex", copy=True)
)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


@app.function(
    image=image,
    gpu="B300",
    timeout=30 * 60,
    volumes={CACHE_MOUNT: cache_volume},
)
def run_dense_gemm_remote(
    m: int = 16384,
    n: int = 16384,
    k: int = 16384,
    warmup: int = 20,
    iterations: int = 200,
    gpu_warmup_seconds: float = 10.0,
    trace: bool = False,
    dump_ir: bool = False,
    implementation: str = "tensor_core",
) -> dict:
    import importlib.metadata
    import os
    import time

    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as cutlass_utils
    import torch
    import transformer_engine
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import make_ptr
    from transformer_engine.pytorch.cpp_extensions import general_gemm

    import cutex
    from cutex.benchmark import cuda_benchmark
    from cutex.kernels.dense_gemm_contract import (
        MMA_SHAPE,
        SCALE_FACTOR_ELEMENTS,
        SF_VECTOR_SIZE,
        TILE_K,
        TILE_M,
        TILE_N,
        dense_gemm_contract,
        dense_gemm_flops,
        validate_dense_gemm_shape,
    )
    from cutex.trace import disabled_iket_metadata, run_iket_profile

    m, n, k = validate_dense_gemm_shape(m, n, k)
    if implementation == "tensor_core":
        from cutex.kernels.dense_gemm import dense_gemm as kernel_fn

        kernel_name = KERNEL_BASE_NAME
        kernel_config = {
            "implementation": implementation,
            "arithmetic": "SM103 tcgen05 block-scaled tensor core",
            "mma_instruction": list(MMA_SHAPE),
            "tile": [TILE_M, TILE_N, TILE_K],
            "threads_per_cta": 128,
            "ab_stages": 4,
        }
    elif implementation == "tma_cuda_core_v2":
        from cutex.kernels.dense_gemm_v2 import (
            SCALE_TILE as V2_SCALE_TILE,
            THREADS as V2_THREADS,
            TILE as V2_TILE,
            TMA_STAGES as V2_TMA_STAGES,
            dense_gemm_v2 as kernel_fn,
        )

        kernel_name = f"{KERNEL_BASE_NAME}_tma_cuda_core_v2"
        kernel_config = {
            "implementation": implementation,
            "arithmetic": "TMA-fed scalar FP32 CUDA Core multiply-add",
            "mma_instruction": None,
            "tile": list(V2_TILE),
            "scale_tma_tile": list(V2_SCALE_TILE),
            "threads_per_cta": V2_THREADS,
            "ab_stages": V2_TMA_STAGES,
            "synchronization": "manual single-stage mbarrier",
            "smem_layout": "unswizzled row-major A/B plus native scale layout",
        }
    elif implementation == "cuda_core_v1":
        from cutex.kernels.dense_gemm_v1 import (
            THREADS as V1_THREADS,
            TILE as V1_TILE,
            dense_gemm_v1 as kernel_fn,
        )

        kernel_name = f"{KERNEL_BASE_NAME}_cuda_core_v1"
        kernel_config = {
            "implementation": implementation,
            "arithmetic": "scalar FP32 CUDA Core multiply-add",
            "mma_instruction": None,
            "tile": list(V1_TILE),
            "threads_per_cta": V1_THREADS,
            "ab_stages": 1,
        }
    elif implementation == "manual_pipeline_v9":
        from cutex.kernels.dense_gemm_v9 import (
            AB_STAGES as V9_AB_STAGES,
            CLUSTER_SHAPE as V9_CLUSTER_SHAPE,
            MMA_INSTRUCTION as V9_MMA_INSTRUCTION,
            MMA_TILE as V9_MMA_TILE,
            THREADS as V9_THREADS,
            dense_gemm_v9 as kernel_fn,
        )

        kernel_name = f"{KERNEL_BASE_NAME}_manual_pipeline_v9"
        kernel_config = {
            "implementation": implementation,
            "arithmetic": "SM103 2CTA tcgen05 block-scaled tensor core",
            "mma_instruction": list(V9_MMA_INSTRUCTION),
            "tile": list(V9_MMA_TILE),
            "threads_per_cta": V9_THREADS,
            "cluster": list(V9_CLUSTER_SHAPE),
            "ab_stages": V9_AB_STAGES,
            "rasterization": "row-major cluster tiles",
            "synchronization": "manual TMA-UMMA and UMMA-epilogue mbarriers",
        }
    elif implementation == "manual_pipeline_v10":
        from cutex.kernels.dense_gemm_v10 import (
            AB_STAGES as V10_AB_STAGES,
            CLUSTER_SHAPE as V10_CLUSTER_SHAPE,
            MMA_INSTRUCTION as V10_MMA_INSTRUCTION,
            MMA_TILE as V10_MMA_TILE,
            THREADS as V10_THREADS,
            dense_gemm_v10 as kernel_fn,
        )

        kernel_name = f"{KERNEL_BASE_NAME}_manual_pipeline_v10"
        kernel_config = {
            "implementation": implementation,
            "arithmetic": "SM103 2CTA tcgen05 block-scaled tensor core",
            "mma_instruction": list(V10_MMA_INSTRUCTION),
            "tile": list(V10_MMA_TILE),
            "threads_per_cta": V10_THREADS,
            "cluster": list(V10_CLUSTER_SHAPE),
            "ab_stages": V10_AB_STAGES,
            "rasterization": "Z-order (Morton) cluster tiles",
            "synchronization": "manual TMA-UMMA and UMMA-epilogue mbarriers",
        }
    elif implementation == "manual_pipeline_v11":
        from cutex.kernels.dense_gemm_v11 import (
            AB_STAGES as V11_AB_STAGES,
            CLUSTER_SHAPE as V11_CLUSTER_SHAPE,
            CLUSTER_SWIZZLE_SIZE as V11_CLUSTER_SWIZZLE_SIZE,
            MMA_INSTRUCTION as V11_MMA_INSTRUCTION,
            MMA_TILE as V11_MMA_TILE,
            THREADS as V11_THREADS,
            dense_gemm_v11 as kernel_fn,
        )

        kernel_name = f"{KERNEL_BASE_NAME}_manual_pipeline_v11"
        kernel_config = {
            "implementation": implementation,
            "arithmetic": "SM103 2CTA tcgen05 block-scaled tensor core",
            "mma_instruction": list(V11_MMA_INSTRUCTION),
            "tile": list(V11_MMA_TILE),
            "threads_per_cta": V11_THREADS,
            "cluster": list(V11_CLUSTER_SHAPE),
            "ab_stages": V11_AB_STAGES,
            "cluster_swizzle_size": V11_CLUSTER_SWIZZLE_SIZE,
            "rasterization": "CuTe layout 8x8 cluster block swizzle",
            "synchronization": "manual TMA-UMMA and UMMA-epilogue mbarriers",
        }
    elif implementation == "manual_pipeline_v12":
        from cutex.kernels.dense_gemm_v12 import (
            AB_STAGES as V12_AB_STAGES,
            ACC_PHYSICAL_STAGES as V12_ACC_PHYSICAL_STAGES,
            CLUSTER_SHAPE as V12_CLUSTER_SHAPE,
            CLUSTER_SWIZZLE_SIZE as V12_CLUSTER_SWIZZLE_SIZE,
            MMA_INSTRUCTION as V12_MMA_INSTRUCTION,
            MMA_TILE as V12_MMA_TILE,
            THREADS as V12_THREADS,
            dense_gemm_v12 as kernel_fn,
        )

        kernel_name = f"{KERNEL_BASE_NAME}_manual_pipeline_v12"
        kernel_config = {
            "implementation": implementation,
            "arithmetic": "SM103 2CTA tcgen05 block-scaled tensor core",
            "mma_instruction": list(V12_MMA_INSTRUCTION),
            "tile": list(V12_MMA_TILE),
            "threads_per_cta": V12_THREADS,
            "cluster": list(V12_CLUSTER_SHAPE),
            "ab_stages": V12_AB_STAGES,
            "physical_accumulator_views": V12_ACC_PHYSICAL_STAGES,
            "cluster_swizzle_size": V12_CLUSTER_SWIZZLE_SIZE,
            "rasterization": "native static persistent 8x8 cluster swizzle",
            "synchronization": "persistent manual mbarriers with early accumulator release",
        }
    else:
        raise ValueError(
            "implementation must be 'tensor_core', 'tma_cuda_core_v2', "
            "'cuda_core_v1', 'manual_pipeline_v9', 'manual_pipeline_v10', "
            "'manual_pipeline_v11', or 'manual_pipeline_v12'"
        )
    if trace and implementation not in {
        "tensor_core",
        "manual_pipeline_v9",
        "manual_pipeline_v10",
        "manual_pipeline_v11",
        "manual_pipeline_v12",
    }:
        raise ValueError(f"IKET worker is not wired for {implementation}")
    is_manual_pipeline = implementation in {
        "manual_pipeline_v9",
        "manual_pipeline_v10",
        "manual_pipeline_v11",
        "manual_pipeline_v12",
    }
    if warmup < 0 or iterations < 1 or gpu_warmup_seconds < 0:
        raise ValueError(
            "warmup must be >= 0, iterations must be >= 1, and GPU warmup must be >= 0"
        )
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
            f"this fixed kernel targets B300/SM103; got {properties.name} "
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
    if dump_ir:
        dump_dir = remote_run_dir / "cute-dsl-dump"
        dump_dir.mkdir(parents=True, exist_ok=True)
        os.environ.update(
            {
                "CUTE_DSL_DUMP_DIR": str(dump_dir),
                "CUTE_DSL_KEEP_PTX": "1",
                "CUTE_DSL_LINEINFO": "1",
            }
        )

    def warm_gpu(duration_seconds: float) -> None:
        if duration_seconds <= 0:
            return
        lhs = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        rhs = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        until = time.perf_counter() + duration_seconds
        while time.perf_counter() < until:
            for _ in range(10):
                torch.mm(lhs, rhs.T)
            torch.cuda.synchronize()
        del lhs, rhs
        torch.cuda.empty_cache()

    warm_gpu(gpu_warmup_seconds)
    torch.manual_seed(20260822)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

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
        if tensor._rowwise_data is None or tensor._rowwise_scale_inv is None:
            raise AssertionError(f"{name} is missing rowwise data or E8M0 scales")

    fp8, sf8 = cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU
    x_ptr = make_ptr(
        fp8, x_q._rowwise_data.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    weight_ptr = make_ptr(
        fp8,
        weight_q._rowwise_data.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    x_scale_ptr = make_ptr(
        sf8,
        x_q._rowwise_scale_inv.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    weight_scale_ptr = make_ptr(
        sf8,
        weight_q._rowwise_scale_inv.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=32,
    )
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    te_output = torch.empty_like(output)
    output_ptr = make_ptr(
        cutlass.BFloat16,
        output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    torch_stream = torch.cuda.current_stream()
    cuda_stream = cuda.CUstream(torch_stream.cuda_stream)

    compile_started = time.perf_counter()
    if implementation == "manual_pipeline_v12":
        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(
            V12_CLUSTER_SHAPE[0] * V12_CLUSTER_SHAPE[1],
            stream=cuda_stream,
        )
        if max_active_clusters < 1:
            raise RuntimeError("CUDA reported no active 2-CTA clusters")
        logical_output_clusters = (m // V12_MMA_TILE[0]) * (
            n // V12_MMA_TILE[1]
        )
        kernel_config["max_active_clusters"] = max_active_clusters
        kernel_config["persistent_grid"] = [
            V12_CLUSTER_SHAPE[0],
            V12_CLUSTER_SHAPE[1],
            max_active_clusters,
        ]
        kernel_config["logical_output_clusters"] = logical_output_clusters
        kernel_config["work_tiles_per_cluster"] = [
            logical_output_clusters // max_active_clusters,
            (logical_output_clusters + max_active_clusters - 1)
            // max_active_clusters,
        ]
        compiled = cutex.compile(
            kernel_fn,
            x_ptr,
            weight_ptr,
            x_scale_ptr,
            weight_scale_ptr,
            output_ptr,
            max_active_clusters,
            cuda_stream,
        )
    else:
        compiled = cutex.compile(
            kernel_fn,
            x_ptr,
            weight_ptr,
            x_scale_ptr,
            weight_scale_ptr,
            output_ptr,
            cuda_stream,
        )
    compile_seconds = time.perf_counter() - compile_started

    def te_fprop():
        general_gemm(
            weight_q,
            x_q,
            out_dtype=torch.bfloat16,
            out=te_output,
            layout="TN",
            use_split_accumulator=True,
        )

    compiled(
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        output_ptr,
        cuda_stream,
    )
    te_fprop()
    torch.cuda.synchronize()
    fp32_reference = torch.mm(x.float(), weight.float().T)
    custom_error = output.float() - fp32_reference
    te_error = te_output.float() - fp32_reference
    custom_relative_l2 = float(
        torch.linalg.vector_norm(custom_error)
        / torch.linalg.vector_norm(fp32_reference)
    )
    te_relative_l2 = float(
        torch.linalg.vector_norm(te_error) / torch.linalg.vector_norm(fp32_reference)
    )
    custom_vs_te = output.float() - te_output.float()
    custom_vs_te_relative_l2 = float(
        torch.linalg.vector_norm(custom_vs_te)
        / torch.linalg.vector_norm(te_output.float())
    )
    if not torch.isfinite(output).all() or custom_relative_l2 >= 0.15:
        raise AssertionError(
            f"MXFP8 correctness failed: relative_l2={custom_relative_l2}"
        )

    kernel_stats = cuda_benchmark(
        compiled,
        x_ptr,
        weight_ptr,
        x_scale_ptr,
        weight_scale_ptr,
        output_ptr,
        cuda_stream,
        warmup=warmup,
        rep=iterations,
        stream=torch_stream,
    )
    te_stats = cuda_benchmark(
        te_fprop, warmup=warmup, rep=iterations, stream=torch_stream
    )
    flop_count = dense_gemm_flops(m, n, k)
    trace_metadata = disabled_iket_metadata()
    if trace:
        trace_cluster = None
        if is_manual_pipeline:
            trace_cluster = (
                (0, 0, 0)
                if implementation == "manual_pipeline_v12"
                else (32, 32, 0)
            )
        iket_result = run_iket_profile(
            remote_run_dir / "iket",
            [
                "dense-gemm",
                "--m",
                str(m),
                "--n",
                str(n),
                "--k",
                str(k),
                "--implementation",
                implementation,
            ],
            instrumented_cluster=trace_cluster,
            max_ts_cnt_per_warp=2048 if is_manual_pipeline else None,
        )
        trace_metadata = iket_result.to_metadata(volume_root=CACHE_MOUNT)
        trace_metadata["workload_shape"] = {"m": m, "n": n, "k": k}

    result = {
        "status": "PASS",
        "run_id": run_id,
        "kernel": kernel_name,
        "contract": dense_gemm_contract(),
        "configuration": {
            "architecture": "sm_103a",
            **kernel_config,
            "scale_vector_size": SF_VECTOR_SIZE,
            "scale_factor_elements_per_operand": SCALE_FACTOR_ELEMENTS,
        },
        "correctness": {
            "all_finite": bool(torch.isfinite(output).all().item()),
            "reference": "FP32 torch.mm on original BF16 inputs",
            "custom_relative_l2_vs_fp32": custom_relative_l2,
            "te_relative_l2_vs_fp32": te_relative_l2,
            "custom_relative_l2_vs_te": custom_vs_te_relative_l2,
            "custom_max_abs_vs_fp32": float(custom_error.abs().max().item()),
            "custom_max_abs_vs_te": float(custom_vs_te.abs().max().item()),
            "output_dtype": str(output.dtype),
        },
        "benchmark": {
            "timer": "CUDA events",
            "gpu_warmup_seconds": gpu_warmup_seconds,
            "warmup": warmup,
            "iterations": iterations,
            "flop_count": flop_count,
            "kernel": {
                **kernel_stats.to_dict(include_samples=False),
                "tflops": flop_count / kernel_stats.median_us / 1e6,
            },
            "transformer_engine": {
                **te_stats.to_dict(include_samples=False),
                "tflops": flop_count / te_stats.median_us / 1e6,
                "backend": "cuBLASLt via Transformer Engine general_gemm",
            },
        },
        "compilation": {
            "location": "modal_remote_container",
            "seconds": compile_seconds,
            "dump_ir": dump_ir,
        },
        "environment": {
            "gpu_name": str(properties.name),
            "compute_capability": f"{properties.major}.{properties.minor}",
            "torch": str(torch.__version__),
            "transformer_engine": str(transformer_engine.__version__),
            "cutex": str(cutex.__version__),
            "nvidia_cutlass_dsl": str(
                importlib.metadata.version("nvidia-cutlass-dsl")
            ),
            "mxfp8_available": bool(available),
            "mxfp8_reason": str(reason),
        },
        "remote_artifacts": {
            "volume": "cutex-autotune-cache",
            "run_directory": str(remote_run_dir),
        },
        "trace": trace_metadata,
    }
    (remote_run_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    cache_volume.commit()
    return result


@app.local_entrypoint()
def main(
    m: int = 16384,
    n: int = 16384,
    k: int = 16384,
    warmup: int = 20,
    iterations: int = 200,
    gpu_warmup_seconds: float = 10.0,
    trace: bool = False,
    dump_ir: bool = False,
    implementation: str = "tensor_core",
):
    result = run_dense_gemm_remote.remote(
        m=m,
        n=n,
        k=k,
        warmup=warmup,
        iterations=iterations,
        gpu_warmup_seconds=gpu_warmup_seconds,
        trace=trace,
        dump_ir=dump_ir,
        implementation=implementation,
    )
    run_dir = LOCAL_ARTIFACT_ROOT / f"{result['run_id']}-{result['kernel']}"
    run_dir.mkdir(parents=True, exist_ok=True)
    from cutex.trace import download_iket_artifacts

    downloaded = download_iket_artifacts(
        cache_volume, result["trace"].get("files", []), run_dir
    )
    result["trace"]["local_files"] = [str(path.resolve()) for path in downloaded]
    result_path = run_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gpu": result["environment"]["gpu_name"],
                "contract": result["contract"],
                "configuration": result["configuration"],
                "correctness": result["correctness"],
                "benchmark": result["benchmark"],
                "trace_files": result["trace"]["local_files"],
                "local_artifact": str(result_path.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
