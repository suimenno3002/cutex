import pytest

from cutex.kernels.dense_gemm_contract import (
    K,
    M,
    N,
    SCALE_FACTOR_ELEMENTS,
    SF_VECTOR_SIZE,
    dense_gemm_contract,
    dense_gemm_flops,
    validate_dense_gemm_shape,
)


def test_dense_gemm_contract_accepts_only_16384_cubed():
    assert validate_dense_gemm_shape(M, N, K) == (16384, 16384, 16384)
    assert dense_gemm_flops() == 2 * 16384**3
    assert SF_VECTOR_SIZE == 32
    assert SCALE_FACTOR_ELEMENTS == 16384 * 16384 // 32


def test_dense_gemm_precision_contract():
    contract = dense_gemm_contract()
    assert contract["inputs"] == "MXFP8 E4M3FN"
    assert contract["multiply"] == "FP8 E4M3FN x E4M3FN"
    assert contract["accumulator"] == "FP32"
    assert contract["fast_accumulation"] is False
    assert contract["output"] == "BF16"
    assert contract["quantization_in_timed_region"] is False


@pytest.mark.parametrize(
    "shape", [(0, 16384, 16384), (128, 256, 128), (16384, 16384, 16383)]
)
def test_dense_gemm_contract_rejects_every_other_shape(shape):
    with pytest.raises(ValueError, match="fixed to M=N=K=16384"):
        validate_dense_gemm_shape(*shape)
