"""Single-launch CuTeDSL workloads executed as children of ``run-iket``."""

from __future__ import annotations

import argparse
import json


def _require_supported_gpu(torch) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("IKET worker requires a CUDA GPU")
    properties = torch.cuda.get_device_properties(0)
    if properties.major < 9:
        raise RuntimeError(
            "CUTLASS IKET requires SM90 or newer; "
            f"got {properties.name} (SM{properties.major}{properties.minor})"
        )


def _run_vector_add(m: int, n: int, copy_bits: int) -> dict:
    import cutlass.cute as cute
    import torch
    from cutlass.cute.runtime import from_dlpack

    from cutex.kernels.vector_add import vector_add

    _require_supported_gpu(torch)
    if m < 1 or n < 1:
        raise ValueError("m and n must both be positive")
    if copy_bits not in {32, 64, 128}:
        raise ValueError("copy_bits must be one of: 32, 64, 128")

    torch.manual_seed(20260822)
    a = torch.randn((m, n), device="cuda", dtype=torch.float32)
    b = torch.randn((m, n), device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)
    a_cute = from_dlpack(a, assumed_align=16).mark_layout_dynamic()
    b_cute = from_dlpack(b, assumed_align=16).mark_layout_dynamic()
    c_cute = from_dlpack(c, assumed_align=16).mark_layout_dynamic()
    compiled = cute.compile(
        vector_add,
        a_cute,
        b_cute,
        c_cute,
        m,
        n,
        copy_bits=copy_bits,
    )
    compiled(a_cute, b_cute, c_cute, m, n)
    torch.cuda.synchronize()
    return {
        "kernel": "vector_add",
        "shape": [m, n],
        "copy_bits": copy_bits,
        "launches": 1,
    }


def _run_dense_gemm(
    m: int, n: int, k: int, implementation: str = "tensor_core"
) -> dict:
    import cutlass
    import cutlass.cute as cute
    import torch
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from cuda.bindings import driver as cuda
    from cutlass.cute.runtime import make_ptr

    from cutex.kernels.dense_gemm_contract import validate_dense_gemm_shape

    if implementation == "tensor_core":
        from cutex.kernels.dense_gemm import dense_gemm as kernel_fn
    elif implementation == "manual_pipeline_v9":
        from cutex.kernels.dense_gemm_v9 import dense_gemm_v9 as kernel_fn
    elif implementation == "manual_pipeline_v10":
        from cutex.kernels.dense_gemm_v10 import dense_gemm_v10 as kernel_fn
    elif implementation == "manual_pipeline_v11":
        from cutex.kernels.dense_gemm_v11 import dense_gemm_v11 as kernel_fn
    else:
        raise ValueError(
            "implementation must be 'tensor_core', 'manual_pipeline_v9', "
            "'manual_pipeline_v10', or 'manual_pipeline_v11'"
        )

    _require_supported_gpu(torch)
    m, n, k = validate_dense_gemm_shape(m, n, k)
    torch.manual_seed(20260822)
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

    def quantize(tensor):
        quantizer = te.MXFP8Quantizer(
            tex.DType.kFloat8E4M3, rowwise=True, columnwise=False
        )
        quantizer.optimize_for_gemm = True
        return quantizer.quantize(tensor)

    a_q, b_q = quantize(a), quantize(b)
    c = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    address_space = cute.AddressSpace.gmem
    a_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        a_q._rowwise_data.data_ptr(),
        address_space,
        assumed_align=16,
    )
    b_ptr = make_ptr(
        cutlass.Float8E4M3FN,
        b_q._rowwise_data.data_ptr(),
        address_space,
        assumed_align=16,
    )
    sfa_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        a_q._rowwise_scale_inv.data_ptr(),
        address_space,
        assumed_align=32,
    )
    sfb_ptr = make_ptr(
        cutlass.Float8E8M0FNU,
        b_q._rowwise_scale_inv.data_ptr(),
        address_space,
        assumed_align=32,
    )
    c_ptr = make_ptr(
        cutlass.BFloat16, c.data_ptr(), address_space, assumed_align=16
    )
    torch_stream = torch.cuda.current_stream()
    cuda_stream = cuda.CUstream(torch_stream.cuda_stream)
    compiled = cute.compile(
        kernel_fn, a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, cuda_stream
    )
    compiled(a_ptr, b_ptr, sfa_ptr, sfb_ptr, c_ptr, cuda_stream)
    torch.cuda.synchronize()
    return {
        "kernel": {
            "tensor_core": "dense_gemm",
            "manual_pipeline_v9": "dense_gemm_v9",
            "manual_pipeline_v10": "dense_gemm_v10",
            "manual_pipeline_v11": "dense_gemm_v11",
        }[implementation],
        "implementation": implementation,
        "shape": {"m": m, "n": n, "k": k},
        "precision": "rowwise MXFP8 E4M3/E8M0, FP32 accumulate, BF16 output",
        "launches": 1,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kernel", required=True)

    vector = subparsers.add_parser("vector-add")
    vector.add_argument("--m", type=int, required=True)
    vector.add_argument("--n", type=int, required=True)
    vector.add_argument("--copy-bits", type=int, required=True)

    gemm = subparsers.add_parser("dense-gemm")
    gemm.add_argument("--m", type=int, required=True)
    gemm.add_argument("--n", type=int, required=True)
    gemm.add_argument("--k", type=int, required=True)
    gemm.add_argument(
        "--implementation",
        choices=(
            "tensor_core",
            "manual_pipeline_v9",
            "manual_pipeline_v10",
            "manual_pipeline_v11",
        ),
        default="tensor_core",
    )

    args = parser.parse_args(argv)
    if args.kernel == "vector-add":
        result = _run_vector_add(args.m, args.n, args.copy_bits)
    else:
        result = _run_dense_gemm(
            args.m, args.n, args.k, implementation=args.implementation
        )
    print(json.dumps({"status": "PASS", **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
