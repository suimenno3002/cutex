"""Minimal tiled MXFP8 GEMM implemented with scalar CUDA Core arithmetic."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

from .dense_gemm_contract import K, M, N, SCALE_FACTOR_ELEMENTS, SF_VECTOR_SIZE


FP8 = cutlass.Float8E4M3FN
SF8 = cutlass.Float8E8M0FNU
F32 = cutlass.Float32
BF16 = cutlass.BFloat16
TILE = (16, 16, 16) # M, N, K
THREADS, AB_STAGES, ACC_STAGES = 256, 1, 0


@cute.struct
class SharedStorage:
    a: cute.struct.Align[cute.struct.MemRange[F32, TILE[0] * TILE[2]], 16]
    b: cute.struct.Align[cute.struct.MemRange[F32, TILE[1] * TILE[2]], 16]


def _scale_index(outer, k):
    """Map logical ``[outer, k // 32]`` into cuBLAS/TE's 128x4 scale tiles."""

    sf_col = k // SF_VECTOR_SIZE
    tile_base = ((sf_col // 4) * 4 + (outer // 128) * (K // 32)) * 128
    tile_offset = (outer % 32) * 16 + ((outer % 128) // 32) * 4 + sf_col % 4
    return tile_base + tile_offset


@cute.kernel
def _dense_gemm_v1_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_sfa: cute.Tensor,
    g_sfb: cute.Tensor,
    g_c: cute.Tensor,
):
    """One 16x16 CTA tile; every thread computes one BF16 output element."""

    tid, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, _ = cute.arch.block_idx()
    local_m, local_n = tid // TILE[1], tid % TILE[1]
    global_m = bid_m * TILE[0] + local_m
    global_n = bid_n * TILE[1] + local_n

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_a = storage.a.get_tensor(
        cute.make_layout((TILE[0], TILE[2]), stride=(TILE[2], 1))
    )
    s_b = storage.b.get_tensor(
        cute.make_layout((TILE[1], TILE[2]), stride=(TILE[2], 1))
    )

    # The same 256 threads that own C also cooperatively load one A and one B
    # value. Values are dequantized to FP32 before entering shared memory.
    load_outer, load_k = tid // TILE[2], tid % TILE[2]
    a_row = bid_m * TILE[0] + load_outer
    b_row = bid_n * TILE[1] + load_outer
    acc = F32(0.0)

    cute.experimental.iket.range_push("dense_gemm_v1_cuda_core")
    cute.experimental.iket.range_push("mainloop")
    for k_tile in cutlass.range(K // TILE[2], unroll=1):
        global_k = k_tile * TILE[2] + load_k
        a = g_a[(a_row, global_k, 0)].to(F32)
        b = g_b[(b_row, global_k, 0)].to(F32)
        sa = g_sfa[_scale_index(a_row, global_k)].to(F32)
        sb = g_sfb[_scale_index(b_row, global_k)].to(F32)
        s_a[(load_outer, load_k)] = a * sa
        s_b[(load_outer, load_k)] = b * sb
        cute.arch.sync_threads()

        # Plain scalar FP32 multiply-add: no cute.gemm, MMA, or tcgen05 path.
        for kk in cutlass.range(TILE[2], unroll_full=True):
            acc = acc + s_a[(local_m, kk)] * s_b[(local_n, kk)]
        cute.arch.sync_threads()
    cute.experimental.iket.range_pop()

    cute.experimental.iket.range_push("epilogue")
    g_c[(global_m, global_n, 0)] = acc.to(BF16)
    cute.experimental.iket.range_pop()
    cute.experimental.iket.range_pop()


@cute.jit
def dense_gemm_v1(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    stream,
):
    """Launch fixed ``BF16 Y = MXFP8 X @ W.T`` using only CUDA Cores.

    A/B remain row-major E4M3FN ``[16384,16384]`` and SFA/SFB remain the
    GEMM-swizzled E8M0FNU scales supplied by Transformer Engine. Each CTA
    dequantizes 16x16 A/B tiles to FP32 shared memory, performs scalar FP32
    multiply-adds, and writes a 16x16 BF16 output tile.
    """

    a = cute.make_tensor(a_ptr, cute.make_layout((M, K, 1), stride=(K, 1, M * K)))
    b = cute.make_tensor(b_ptr, cute.make_layout((N, K, 1), stride=(K, 1, N * K)))
    c = cute.make_tensor(c_ptr, cute.make_layout((M, N, 1), stride=(N, 1, M * N)))
    sfa = cute.make_tensor(
        sfa_ptr, cute.make_layout((SCALE_FACTOR_ELEMENTS,), stride=(1,))
    )
    sfb = cute.make_tensor(
        sfb_ptr, cute.make_layout((SCALE_FACTOR_ELEMENTS,), stride=(1,))
    )
    _dense_gemm_v1_kernel(a, b, sfa, sfb, c).launch(
        grid=(M // TILE[0], N // TILE[1], 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


__all__ = [
    "AB_STAGES",
    "ACC_STAGES",
    "BF16",
    "F32",
    "FP8",
    "SF8",
    "SF_VECTOR_SIZE",
    "THREADS",
    "TILE",
    "dense_gemm_v1",
]
