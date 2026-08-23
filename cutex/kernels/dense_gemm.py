# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Fixed 16384^3 B300 MXFP8 Fprop: rowwise E4M3/E8M0 -> FP32 -> BF16."""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05

from .dense_gemm_contract import (
    K,
    M,
    MMA_SHAPE,
    N,
    SF_VECTOR_SIZE,
    TILE_K,
    TILE_M,
    TILE_N,
)


FP8 = cutlass.Float8E4M3FN
SF8 = cutlass.Float8E8M0FNU
F32 = cutlass.Float32
BF16 = cutlass.BFloat16
TILE = (TILE_M, TILE_N, TILE_K)
THREADS, AB_STAGES, ACC_STAGES = 128, 4, 1


@cute.struct
class SharedStorage:
    ab: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem: cutlass.Int32


@cute.kernel
def _dense_gemm_kernel(
    mma: cute.TiledMma,
    tma_a: cute.CopyAtom,
    g_a: cute.Tensor,
    tma_b: cute.CopyAtom,
    g_b: cute.Tensor,
    tma_sfa: cute.CopyAtom,
    g_sfa: cute.Tensor,
    tma_sfb: cute.CopyAtom,
    g_sfb: cute.Tensor,
    g_c: cute.Tensor,
    s_layout_a: cute.ComposedLayout,
    s_layout_b: cute.ComposedLayout,
    s_layout_sfa: cute.Layout,
    s_layout_sfb: cute.Layout,
    tma_bytes: cutlass.Constexpr,
):
    """一个 CTA 计算一个 128x256 输出 tile，并沿 K=16384 做 FP32 累加。"""

    cute.experimental.iket.range_push("dense_gemm")
    cute.experimental.iket.range_push("prologue")
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tid, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, bid_l = cute.arch.block_idx()
    tile_coord = (bid_m // cute.size(mma.thr_id.shape), bid_n, bid_l)

    # A/B 与 scale 都先进入多级 SMEM；FP32 accumulator 和 MMA scale 位于 TMEM。
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_a = smem.allocate_tensor(
        element_type=FP8,
        layout=s_layout_a.outer,
        byte_alignment=128,
        swizzle=s_layout_a.inner,
    )
    s_b = smem.allocate_tensor(
        element_type=FP8,
        layout=s_layout_b.outer,
        byte_alignment=128,
        swizzle=s_layout_b.inner,
    )
    s_sfa = smem.allocate_tensor(
        element_type=SF8, layout=s_layout_sfa, byte_alignment=128
    )
    s_sfb = smem.allocate_tensor(
        element_type=SF8, layout=s_layout_sfb, byte_alignment=128
    )

    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        barrier_storage=storage.ab.data_ptr(),
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
        tx_count=tma_bytes,
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        barrier_storage=storage.acc.data_ptr(),
        num_stages=ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, THREADS),
    ).make_participants()

    # 保留完整的逻辑 M/N/K 坐标；scale tensor 与 A/B 共享坐标，只是物理上已 swizzle。
    l_a = cute.local_tile(g_a, cute.slice_(TILE, (None, 0, None)), (None, None, None))
    l_b = cute.local_tile(g_b, cute.slice_(TILE, (0, None, None)), (None, None, None))
    l_sfa = cute.local_tile(
        g_sfa, cute.slice_(TILE, (None, 0, None)), (None, None, None)
    )
    l_sfb = cute.local_tile(
        g_sfb, cute.slice_(TILE, (0, None, None)), (None, None, None)
    )
    l_c = cute.local_tile(g_c, cute.slice_(TILE, (None, None, 0)), (None, None, None))
    k_tiles = cute.size(l_a, mode=[3])

    mma_thr = mma.get_slice(0)
    mma_g_a, mma_g_b = mma_thr.partition_A(l_a), mma_thr.partition_B(l_b)
    mma_g_sfa, mma_g_sfb = mma_thr.partition_A(l_sfa), mma_thr.partition_B(l_sfb)
    mma_g_c = mma_thr.partition_C(l_c)

    # TMA partition 把同一 CTA 的四个全局 K tiles 映射到同一个 pipeline stage。
    tma_layout = cute.make_layout(1)
    tma_s_a, tma_g_a = cpasync.tma_partition(
        tma_a,
        0,
        tma_layout,
        cute.group_modes(s_a, 0, 3),
        cute.group_modes(mma_g_a, 0, 3),
    )
    tma_s_b, tma_g_b = cpasync.tma_partition(
        tma_b,
        0,
        tma_layout,
        cute.group_modes(s_b, 0, 3),
        cute.group_modes(mma_g_b, 0, 3),
    )
    tma_s_sfa, tma_g_sfa = cpasync.tma_partition(
        tma_sfa,
        0,
        tma_layout,
        cute.group_modes(s_sfa, 0, 3),
        cute.group_modes(mma_g_sfa, 0, 3),
    )
    tma_s_sfb, tma_g_sfb = cpasync.tma_partition(
        tma_sfb,
        0,
        tma_layout,
        cute.group_modes(s_sfb, 0, 3),
        cute.group_modes(mma_g_sfb, 0, 3),
    )
    tma_s_sfa, tma_g_sfa = cute.filter_zeros(tma_s_sfa), cute.filter_zeros(tma_g_sfa)
    tma_s_sfb, tma_g_sfb = cute.filter_zeros(tma_s_sfb), cute.filter_zeros(tma_g_sfb)

    r_a, r_b = mma.make_fragment_A(s_a), mma.make_fragment_B(s_b)
    fake_acc = mma.make_fragment_C(mma.partition_shape_C(TILE[:2]))
    tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=THREADS)
    tmem = utils.TmemAllocator(storage.tmem.ptr, barrier_for_retrieve=tmem_barrier)
    tmem.allocate(512)
    tmem.wait_for_alloc()
    acc_ptr = tmem.retrieve_ptr(F32)
    acc = cute.make_tensor(acc_ptr, fake_acc.layout)

    # Block-scaled MMA 不从普通寄存器读 scale：每个 stage 到齐后先 SMEM -> TMEM。
    sfa_ptr = cute.recast_ptr(
        acc_ptr + tcgen05.find_tmem_tensor_col_offset(acc), dtype=SF8
    )
    sfa_tmem_layout = blockscaled_utils.make_tmem_layout_sfa(
        mma,
        TILE,
        SF_VECTOR_SIZE,
        cute.slice_(s_layout_sfa, (None, None, None, 0)),
    )
    t_sfa = cute.make_tensor(sfa_ptr, sfa_tmem_layout)
    sfb_ptr = cute.recast_ptr(
        acc_ptr
        + tcgen05.find_tmem_tensor_col_offset(acc)
        + tcgen05.find_tmem_tensor_col_offset(t_sfa),
        dtype=SF8,
    )
    sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
        mma,
        TILE,
        SF_VECTOR_SIZE,
        cute.slice_(s_layout_sfb, (None, None, None, 0)),
    )
    t_sfb = cute.make_tensor(sfb_ptr, sfb_tmem_layout)

    s2t_atom = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), SF8
    )
    s_sfa_compact, t_sfa_compact = cute.filter_zeros(s_sfa), cute.filter_zeros(t_sfa)
    copy_sfa = tcgen05.make_s2t_copy(s2t_atom, t_sfa_compact)
    copy_sfa_thr = copy_sfa.get_slice(0)
    copy_sfa_src = tcgen05.get_s2t_smem_desc_tensor(
        copy_sfa, copy_sfa_thr.partition_S(s_sfa_compact)
    )
    copy_sfa_dst = copy_sfa_thr.partition_D(t_sfa_compact)

    s_sfb_compact, t_sfb_compact = cute.filter_zeros(s_sfb), cute.filter_zeros(t_sfb)
    copy_sfb = tcgen05.make_s2t_copy(s2t_atom, t_sfb_compact)
    copy_sfb_thr = copy_sfb.get_slice(0)
    copy_sfb_src = tcgen05.get_s2t_smem_desc_tensor(
        copy_sfb, copy_sfb_thr.partition_S(s_sfb_compact)
    )
    copy_sfb_dst = copy_sfb_thr.partition_D(t_sfb_compact)

    tma_g_a = tma_g_a[(None, tile_coord[0], None, tile_coord[2])]
    tma_g_b = tma_g_b[(None, tile_coord[1], None, tile_coord[2])]
    tma_g_sfa = tma_g_sfa[(None, tile_coord[0], None, tile_coord[2])]
    tma_g_sfb = tma_g_sfb[(None, tile_coord[1], None, tile_coord[2])]
    cute.experimental.iket.range_pop()

    # warp 0 串行发射异步 TMA/S2T/MMA；硬件 pipeline 与四级 stage 负责重叠。
    if warp == 0:
        cute.experimental.iket.range_push("mainloop")
        acc_empty = acc_producer.acquire_and_advance()
        mma.set(tcgen05.Field.ACCUMULATE, False)
        for kt in cutlass.range(k_tiles, prefetch_stages=AB_STAGES - 2):
            cute.experimental.iket.range_push("k_tile", kt)
            stage_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_a,
                tma_g_a[(None, stage_empty.count)],
                tma_s_a[(None, stage_empty.index)],
                tma_bar_ptr=stage_empty.barrier,
            )
            cute.copy(
                tma_b,
                tma_g_b[(None, stage_empty.count)],
                tma_s_b[(None, stage_empty.index)],
                tma_bar_ptr=stage_empty.barrier,
            )
            cute.copy(
                tma_sfa,
                tma_g_sfa[(None, stage_empty.count)],
                tma_s_sfa[(None, stage_empty.index)],
                tma_bar_ptr=stage_empty.barrier,
            )
            cute.copy(
                tma_sfb,
                tma_g_sfb[(None, stage_empty.count)],
                tma_s_sfb[(None, stage_empty.index)],
                tma_bar_ptr=stage_empty.barrier,
            )
            stage_full = ab_consumer.wait_and_advance()

            stage = (None, None, None, None, stage_full.index)
            cute.copy(copy_sfa, copy_sfa_src[stage], copy_sfa_dst)
            cute.copy(copy_sfb, copy_sfb_src[stage], copy_sfb_dst)

            # 一个 128-K CTA tile 含四条 K=32 MMA；首条覆盖 accumulator，之后累加。
            for kb in cutlass.range(cute.size(r_a, mode=[2]), unroll_full=True):
                mma.set(tcgen05.Field.SFA, t_sfa[(None, None, kb)].iterator)
                mma.set(tcgen05.Field.SFB, t_sfb[(None, None, kb)].iterator)
                coord = (None, None, kb, stage_full.index)
                cute.gemm(mma, acc, r_a[coord], r_b[coord], acc)
                mma.set(tcgen05.Field.ACCUMULATE, True)
            stage_full.release()
            cute.experimental.iket.range_pop()
        acc_empty.commit()
        cute.experimental.iket.range_pop()

    # 128 个线程共同把 FP32 TMEM tile 转成 BF16 并写回 row-major Y。
    cute.experimental.iket.range_push("epilogue")
    t2r_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE), F32
    )
    t2r = tcgen05.make_tmem_copy(t2r_atom, acc)
    t2r_thr = t2r.get_slice(tid)
    t_acc = t2r_thr.partition_S(acc)
    t_out = t2r_thr.partition_D(mma_g_c)
    r_acc = cute.make_rmem_tensor(t_out[None, None, None, None, 0, 0, 0].shape, F32)
    r_out = cute.make_rmem_tensor(t_out[None, None, None, None, 0, 0, 0].shape, BF16)
    out = t_out[(None, None, None, None, *tile_coord)]

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    cute.copy(t2r, t_acc, r_acc)
    r_out.store(r_acc.load().to(BF16))
    cute.copy(cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF16), r_out, out)
    acc_full.release()
    cute.arch.barrier()
    tmem.free(acc_ptr)
    cute.experimental.iket.range_pop()
    cute.experimental.iket.range_pop()


@cute.jit
def dense_gemm(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    stream,
):
    """Launch fixed ``BF16 Y = MXFP8 X @ W.T`` on pre-quantized inputs.

    A/B are row-major E4M3FN ``[16384,16384]``. SFA/SFB contain E8M0FNU
    dequantization scales in tcgen05 Swizzle32x4x4 layout, one per 32 K values.
    Quantization is outside this kernel; MMA multiplication is FP8, the single
    TMEM accumulator is FP32, and output is BF16.
    """

    a = cute.make_tensor(a_ptr, cute.make_layout((M, K, 1), stride=(K, 1, M * K)))
    b = cute.make_tensor(b_ptr, cute.make_layout((N, K, 1), stride=(K, 1, N * K)))
    c = cute.make_tensor(c_ptr, cute.make_layout((M, N, 1), stride=(N, 1, M * N)))
    sfa = cute.make_tensor(
        sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VECTOR_SIZE)
    )
    sfb = cute.make_tensor(
        sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VECTOR_SIZE)
    )

    mma = cute.make_tiled_mma(
        tcgen05.MmaMXF8F6F4Op(
            FP8,
            FP8,
            MMA_SHAPE,
            tcgen05.CtaGroup.ONE,
            tcgen05.OperandSource.SMEM,
            OperandMajorMode.K,
            OperandMajorMode.K,
        )
    )
    s_layout_a = sm100_utils.make_smem_layout_a(mma, TILE, FP8, AB_STAGES)
    s_layout_b = sm100_utils.make_smem_layout_b(mma, TILE, FP8, AB_STAGES)
    s_layout_sfa = blockscaled_utils.make_smem_layout_sfa(
        mma, TILE, SF_VECTOR_SIZE, AB_STAGES
    )
    s_layout_sfb = blockscaled_utils.make_smem_layout_sfb(
        mma, TILE, SF_VECTOR_SIZE, AB_STAGES
    )
    cluster = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma.thr_id.shape,))
    tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
    tma_a, g_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        a,
        cute.slice_(s_layout_a, (None, None, None, 0)),
        TILE,
        mma,
        cluster.shape,
    )
    tma_b, g_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        b,
        cute.slice_(s_layout_b, (None, None, None, 0)),
        TILE,
        mma,
        cluster.shape,
    )
    tma_sfa, g_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        sfa,
        cute.slice_(s_layout_sfa, (None, None, None, 0)),
        TILE,
        mma,
        cluster.shape,
        internal_type=cutlass.Int16,
    )
    tma_sfb, g_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        sfb,
        cute.slice_(s_layout_sfb, (None, None, None, 0)),
        TILE,
        mma,
        cluster.shape,
        internal_type=cutlass.Int16,
    )
    tma_bytes = (
        cute.size_in_bytes(FP8, cute.slice_(s_layout_a, (None, None, None, 0)))
        + cute.size_in_bytes(FP8, cute.slice_(s_layout_b, (None, None, None, 0)))
        + cute.size_in_bytes(SF8, cute.slice_(s_layout_sfa, (None, None, None, 0)))
        + cute.size_in_bytes(SF8, cute.slice_(s_layout_sfb, (None, None, None, 0)))
    ) * cute.size(mma.thr_id.shape)

    _dense_gemm_kernel(
        mma,
        tma_a,
        g_a,
        tma_b,
        g_b,
        tma_sfa,
        g_sfa,
        tma_sfb,
        g_sfb,
        c,
        s_layout_a,
        s_layout_b,
        s_layout_sfa,
        s_layout_sfb,
        tma_bytes,
    ).launch(
        grid=(M // TILE_M, N // TILE_N, 1),
        block=(THREADS, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
    )


__all__ = ["dense_gemm"]
