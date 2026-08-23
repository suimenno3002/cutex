"""Fixed contract for the B300 MXFP8 Fprop kernel."""

from __future__ import annotations


M = N = K = 16384
PROBLEM_SHAPE = (M, N, K)

TILE_M, TILE_N, TILE_K = 128, 256, 128
MMA_SHAPE = (128, 256, 32)

SF_VECTOR_SIZE = 32
SCALE_FACTOR_ELEMENTS = M * K // SF_VECTOR_SIZE
SCALE_FACTOR_LOGICAL_SHAPE = (M, K // SF_VECTOR_SIZE)

OPERATION = "Y[M,N] = X[M,K] @ W[N,K].T"
INPUT_FORMAT = "MXFP8 E4M3FN"
SCALE_FORMAT = "E8M0FNU, one scale per 32 contiguous K values"
SCALE_LAYOUT = "tcgen05 Swizzle32x4x4"
MULTIPLY_FORMAT = "FP8 E4M3FN x E4M3FN"
ACCUMULATOR_FORMAT = "FP32"
OUTPUT_FORMAT = "BF16"


def validate_dense_gemm_shape(m: int, n: int, k: int) -> tuple[int, int, int]:
    """Accept only the benchmark shape this deliberately fixed kernel targets."""

    shape = (int(m), int(n), int(k))
    if shape != PROBLEM_SHAPE:
        raise ValueError(
            f"dense_gemm is fixed to M=N=K=16384; got M={shape[0]}, "
            f"N={shape[1]}, K={shape[2]}"
        )
    return shape


def dense_gemm_flops(m: int = M, n: int = N, k: int = K) -> int:
    """Return the conventional two-FLOP count for ``X @ W.T``."""

    m, n, k = validate_dense_gemm_shape(m, n, k)
    return 2 * m * n * k


def dense_gemm_contract() -> dict[str, object]:
    """Return a serializable precision and storage contract for runners."""

    return {
        "operation": OPERATION,
        "shape": {"m": M, "n": N, "k": K},
        "inputs": INPUT_FORMAT,
        "scales": SCALE_FORMAT,
        "scale_layout": SCALE_LAYOUT,
        "multiply": MULTIPLY_FORMAT,
        "accumulator": ACCUMULATOR_FORMAT,
        "fast_accumulation": False,
        "output": OUTPUT_FORMAT,
        "quantization_in_timed_region": False,
    }


__all__ = [
    "ACCUMULATOR_FORMAT",
    "INPUT_FORMAT",
    "K",
    "M",
    "MMA_SHAPE",
    "MULTIPLY_FORMAT",
    "N",
    "OPERATION",
    "OUTPUT_FORMAT",
    "PROBLEM_SHAPE",
    "SCALE_FACTOR_ELEMENTS",
    "SCALE_FACTOR_LOGICAL_SHAPE",
    "SCALE_FORMAT",
    "SCALE_LAYOUT",
    "SF_VECTOR_SIZE",
    "TILE_K",
    "TILE_M",
    "TILE_N",
    "dense_gemm_contract",
    "dense_gemm_flops",
    "validate_dense_gemm_shape",
]
