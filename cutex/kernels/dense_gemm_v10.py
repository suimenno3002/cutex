"""2SM warp-specialized MXFP8 GEMM with an explicit mbarrier schedule.

The kernel deliberately does not use ``PipelineTmaUmma`` or
``PipelineUmmaAsync``.  Their full/empty barrier protocol is expanded below so
that every stage index, phase transition, transaction count, DSM destination,
and ``tcgen05.commit`` is visible in this file.

Target dataflow for one non-persistent CTA pair::

    TMA warp:       GMEM A/B/SFA/SFB -> six-stage SMEM ring
                                      |
    MMA warp:       wait -> SFA/SFB SMEM->TMEM -> 2CTA MXFP8 MMA -> FP32 TMEM
                                                                   |
    epilogue warps:                      wait -> TMEM->RMEM -> BF16 -> GMEM C

Each ``(2, 1, 1)`` cluster owns exactly one pair-wide ``256 x 256`` output
tile.  There is no persistent tile scheduler and no loop over output tiles;
only the reduction over K is pipelined.

Design references:

* https://zhuanlan.zhihu.com/p/2007400131314595305
* https://zhuanlan.zhihu.com/p/2007431045436442536
* CUTLASS v4.7.0 ``dense_blockscaled_gemm_persistent.py`` (mainloop APIs)
* CUTLASS v4.7.0 ``nvfp4_gemm_1.py`` (non-persistent 2CTA launch/layout)
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.pipeline import NamedBarrier

from .dense_gemm_contract import K, M, N, SF_VECTOR_SIZE


FP8 = cutlass.Float8E4M3FN
SF8 = cutlass.Float8E8M0FNU
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

# Pair-wide MMA tile.  CtaGroup.TWO splits its M mode evenly, so one CTA owns
# a 128x256 half tile while the two-CTA pair cooperatively issues 256x256x32
# block-scaled MMA instructions.  TILE_K=128 means four K=32 instructions per
# SMEM stage.
MMA_INSTRUCTION = (256, 256, 32)
MMA_TILE = (256, 256, 128)
CTA_TILE = (128, 256, 128)

# SFB cannot use the two-CTA SF partition directly in CUTLASS 4.7.0.  Follow
# the official block-scaled examples and build a companion one-CTA tiled MMA
# solely for SFB partition/TMA construction.
SFB_MMA_TILE = (128, 256, 128)

CLUSTER_SHAPE = (2, 1, 1)
# Per CTA and per stage this candidate needs about 16 KiB A + 16 KiB B +
# 0.5 KiB SFA + 1 KiB SFB.  Four stages use 134 KiB for tensor payloads,
# leaving 94 KiB of the B300 228 KiB/SM budget before barrier/TMEM allocator
# metadata and alignment.
AB_STAGES, ACC_STAGES = 6, 1

# Z-order (Morton) cluster rasterization.  The regular grid launches clusters in
# row-major cid order, so each ~74-cluster wave is a 64 x 1-2 column sweep that
# re-reads the whole A panel (or B panel) every wave.  Measured in the AB6 IKET
# profile that puts ~15 GB of read traffic at ~65% of B300 HBM bandwidth, which
# is what turns TMA completion latency into a 1.2-2.6 us variable.  Remapping the
# cluster grid tile via a Z-order curve keeps any launch window spatially compact
# (16 x 10 clusters instead of 64 x 2), cutting unique-bytes-per-wave ~2.4x.
# 6 bits per axis because the cluster tile grid is 64 x 64 (16384 / 256); the
# rank is the cluster's row-major cid and decoding it interleaves the two axes.
ZORDER_BITS = 6


def _zorder_decode(rank: int) -> tuple[int, int]:
    """De-interleave a Z-order rank into 64 x 64 cluster tile coordinates."""
    m = n = 0
    for bit in range(ZORDER_BITS):
        m |= ((rank >> (2 * bit)) & 1) << bit
        n |= ((rank >> (2 * bit + 1)) & 1) << bit
    return m, n

# Four warps drain each CTA's half accumulator.  Separate producer and consumer
# warps are what allow TMA and tcgen05.mma to advance different ring stages.
EPILOGUE_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
THREADS = 6 * 32
K_TILES = K // MMA_TILE[2]
TMEM_COLUMNS = 512
# CUTLASS DSL 4.7.0's run-iket does not yet expose --enabled-cluster.  Emit
# events only from this steady-state cluster in the 64x64 regular cluster grid.
IKET_CLUSTER = (32, 32, 0)


@cute.struct
class SharedStorage:
    """Manual full/empty mbarriers plus TMEM allocator metadata."""

    # First STAGES entries are full barriers; the next STAGES are empty.
    ab_mbar: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def _dense_gemm_v10_kernel(
    mma: cute.TiledMma,
    mma_sfb: cute.TiledMma,
    tma_a: cute.CopyAtom,
    g_a: cute.Tensor,
    tma_b: cute.CopyAtom,
    g_b: cute.Tensor,
    tma_sfa: cute.CopyAtom,
    g_sfa: cute.Tensor,
    tma_sfb: cute.CopyAtom,
    g_sfb: cute.Tensor,
    g_c: cute.Tensor,
    cluster_layout_vmnk: cute.Layout,
    cluster_layout_sfb_vmnk: cute.Layout,
    s_layout_a: cute.ComposedLayout,
    s_layout_b: cute.ComposedLayout,
    s_layout_sfa: cute.Layout,
    s_layout_sfb: cute.Layout,
    tma_bytes: cutlass.Constexpr,
):
    """Compute one pair-wide output tile with a hand-written async schedule."""

    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tid, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, bid_l = cute.arch.block_idx()
    cluster_grid_m = M // MMA_TILE[0]
    cluster_grid_n = N // MMA_TILE[1]
    cluster_linear = bid_m // CLUSTER_SHAPE[0] + cluster_grid_m * (
        bid_n + cluster_grid_n * bid_l
    )
    iket_cluster_linear = IKET_CLUSTER[0] + cluster_grid_m * (
        IKET_CLUSTER[1] + cluster_grid_n * IKET_CLUSTER[2]
    )
    is_iket_cluster = cluster_linear == iket_cluster_linear
    if is_iket_cluster:
        cute.experimental.iket.range_push("dense_gemm_v10")
        cute.experimental.iket.range_push("v10_prologue")

    # Z-order tile remap.  ``cluster_linear`` is the row-major cid the scheduler
    # launched (in the 64 x 64 cluster tile grid), and is computed from the
    # physical block index above.  We change only which output tile this cluster
    # computes: the Z-order curve visits the 64 x 64 grid in a spatially compact
    # order, so consecutive cids stay close and any ~74-cluster launch window is
    # a square footprint instead of a whole 64 x 1-2 column sweep.  Decoding the
    # cid as an interleaved 6-bit x 6-bit rank gives the tile coordinates that
    # the TMA paths and mma slice consume below.  Because both CTAs of the
    # ``(2, 1, 1)`` cluster share the same cid, a pair always keeps the same
    # tile and only differs in the half-M selector.
    tile_m, tile_n = _zorder_decode(cluster_linear)

    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
    cta_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)

    mma_tile_coord_v = bid_m % cute.size(cluster_layout_vmnk, mode=[0])
    mma_tile_coord_mnl = (
        tile_m,
        tile_n,
        bid_l,
    )
    is_leader_cta = mma_tile_coord_v == 0

    # AB_STAGES independent SMEM slots form the A/B/SFA/SFB ring.
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

    # Manual barrier layout.  Each slot has a full generation and an empty
    # generation.  Producer phases start at 1 (an initialized empty barrier is
    # immediately reusable); consumer phases start at 0.
    ab_full_mbar = storage.ab_mbar.data_ptr()
    ab_empty_mbar = ab_full_mbar + AB_STAGES
    acc_full_mbar = storage.acc_mbar.data_ptr()
    acc_empty_mbar = acc_full_mbar + ACC_STAGES

    num_ab_empty_arrivals = (
        cute.size(cluster_layout_vmnk, mode=[1])
        + cute.size(cluster_layout_vmnk, mode=[2])
        - 1
    )
    # One elected lane in every epilogue warp releases the accumulator.  The
    # pair leader therefore expects four arrivals from each of the two CTAs.
    num_acc_empty_arrivals = cute.size(cluster_layout_vmnk, mode=[0]) * len(
        EPILOGUE_WARPS
    )

    if warp == EPILOGUE_WARPS[0]:
        with cute.arch.elect_one():
            for stage in range(AB_STAGES):
                cute.arch.mbarrier_init(ab_full_mbar + stage, 1)
                cute.arch.mbarrier_init(
                    ab_empty_mbar + stage, num_ab_empty_arrivals
                )
            for stage in range(ACC_STAGES):
                cute.arch.mbarrier_init(acc_full_mbar + stage, 1)
                cute.arch.mbarrier_init(
                    acc_empty_mbar + stage, num_acc_empty_arrivals
                )

    # TMEM allocation is synchronized only across the MMA warp and four
    # epilogue warps.  The TMA warp never touches TMEM.
    tmem_alloc_barrier = NamedBarrier(
        barrier_id=1,
        num_threads=(len(EPILOGUE_WARPS) + 1) * 32,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_alloc_barrier,
        allocator_warp_id=EPILOGUE_WARPS[0],
        is_two_cta=True,
        two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
    )

    # Publish every local mbarrier initialization.  cluster_arrive is
    # non-blocking, so tensor partitioning below overlaps the cluster handshake.
    cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive()

    # Build the four logical tiles.  The 2CTA MMA slice selects this CTA's half
    # of A/C; the companion 1CTA MMA is used only for the SFB partition.
    l_a = cute.local_tile(
        g_a, cute.slice_(MMA_TILE, (None, 0, None)), (None, None, None)
    )
    l_b = cute.local_tile(
        g_b, cute.slice_(MMA_TILE, (0, None, None)), (None, None, None)
    )
    l_sfa = cute.local_tile(
        g_sfa, cute.slice_(MMA_TILE, (None, 0, None)), (None, None, None)
    )
    l_sfb = cute.local_tile(
        g_sfb, cute.slice_(MMA_TILE, (0, None, None)), (None, None, None)
    )
    l_c = cute.local_tile(
        g_c, cute.slice_(MMA_TILE, (None, None, 0)), (None, None, None)
    )

    mma_thr = mma.get_slice(mma_tile_coord_v)
    mma_sfb_thr = mma_sfb.get_slice(mma_tile_coord_v)
    mma_g_a = mma_thr.partition_A(l_a)
    mma_g_b = mma_thr.partition_B(l_b)
    mma_g_sfa = mma_thr.partition_A(l_sfa)
    mma_g_sfb = mma_sfb_thr.partition_B(l_sfb)
    mma_g_c = mma_thr.partition_C(l_c)

    # Pair each staged SMEM tensor with its TMA coordinate tensor.
    tma_s_a, tma_g_a = cpasync.tma_partition(
        tma_a,
        cta_coord_vmnk[2],
        cute.make_layout(cute.size(cluster_layout_vmnk, mode=[2])),
        cute.group_modes(s_a, 0, 3),
        cute.group_modes(mma_g_a, 0, 3),
    )
    tma_s_b, tma_g_b = cpasync.tma_partition(
        tma_b,
        cta_coord_vmnk[1],
        cute.make_layout(cute.size(cluster_layout_vmnk, mode=[1])),
        cute.group_modes(s_b, 0, 3),
        cute.group_modes(mma_g_b, 0, 3),
    )
    tma_s_sfa, tma_g_sfa = cpasync.tma_partition(
        tma_sfa,
        cta_coord_vmnk[2],
        cute.make_layout(cute.size(cluster_layout_vmnk, mode=[2])),
        cute.group_modes(s_sfa, 0, 3),
        cute.group_modes(mma_g_sfa, 0, 3),
    )
    tma_s_sfa = cute.filter_zeros(tma_s_sfa)
    tma_g_sfa = cute.filter_zeros(tma_g_sfa)

    sfb_cta_layout = cute.make_layout(
        cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
    )
    tma_s_sfb, tma_g_sfb = cpasync.tma_partition(
        tma_sfb,
        cta_coord_sfb_vmnk[1],
        sfb_cta_layout,
        cute.group_modes(s_sfb, 0, 3),
        cute.group_modes(mma_g_sfb, 0, 3),
    )
    tma_s_sfb = cute.filter_zeros(tma_s_sfb)
    tma_g_sfb = cute.filter_zeros(tma_g_sfb)

    # SMEM descriptors for tcgen05.mma and the logical FP32 accumulator layout.
    r_a = mma.make_fragment_A(s_a)
    r_b = mma.make_fragment_B(s_b)
    acc_shape = mma.partition_shape_C(MMA_TILE[:2])
    fake_acc = mma.make_fragment_C(acc_shape)

    tma_g_a = tma_g_a[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
    tma_g_b = tma_g_b[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
    tma_g_sfa = tma_g_sfa[
        (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
    ]
    tma_g_sfb = tma_g_sfb[
        (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
    ]

    # TMA multicast masks and the inverse mask used by tcgen05.commit to mark
    # every producer's local SMEM stage empty again.
    a_full_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, cta_coord_vmnk, mcast_mode=2
    )
    b_full_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, cta_coord_vmnk, mcast_mode=1
    )
    sfa_full_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_vmnk, cta_coord_vmnk, mcast_mode=2
    )
    sfb_full_mask = cpasync.create_tma_multicast_mask(
        cluster_layout_sfb_vmnk, cta_coord_sfb_vmnk, mcast_mode=1
    )
    peer_coord_vmnk = (cta_coord_vmnk[0] ^ 1, *cta_coord_vmnk[1:])
    ab_empty_mask = (
        a_full_mask
        | b_full_mask
        | cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, peer_coord_vmnk, mcast_mode=2
        )
        | cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, peer_coord_vmnk, mcast_mode=1
        )
    )

    # One 2CTA MMA completion makes the accumulator visible in both CTAs.  One
    # lane per epilogue warp releases the empty barrier in the pair's leader.
    acc_full_mask = cute.make_layout_image_mask(
        cluster_layout_vmnk, cta_coord_vmnk, mode=0
    )
    acc_empty_dst_rank = cta_rank // 2 * 2

    cute.arch.cluster_wait()

    # Keep the allocator's compile-time column bookkeeping in the outer region
    # so the later free() sees the same 512-column allocation.  allocate()
    # itself predicates the hardware instruction on EPILOGUE_WARPS[0].
    tmem.allocate(TMEM_COLUMNS)
    if is_iket_cluster:
        cute.experimental.iket.range_pop()  # v10_prologue

    # TMA producer warp: wait for each slot's empty generation, arm the leader
    # full barrier with the exact byte count, then issue four asynchronous loads.
    if warp == TMA_WARP:
        if is_iket_cluster:
            cute.experimental.iket.range_push("tma_main")
        for k_tile in cutlass.range(K_TILES, unroll=1):
            if is_iket_cluster:
                cute.experimental.iket.range_push("tma_k_tile", k_tile)
            stage = k_tile % AB_STAGES
            generation = (k_tile // AB_STAGES) % 2
            empty_phase = generation ^ 1
            if is_iket_cluster:
                cute.experimental.iket.range_push("tma_wait_empty")
            cute.arch.mbarrier_wait(ab_empty_mbar + stage, empty_phase)
            if is_iket_cluster:
                cute.experimental.iket.range_pop()

            if is_iket_cluster:
                cute.experimental.iket.range_push("tma_issue")
            if is_leader_cta:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        ab_full_mbar + stage, tma_bytes
                    )

            cute.copy(
                tma_a,
                tma_g_a[(None, k_tile)],
                tma_s_a[(None, stage)],
                tma_bar_ptr=ab_full_mbar + stage,
                mcast_mask=a_full_mask,
            )
            cute.copy(
                tma_b,
                tma_g_b[(None, k_tile)],
                tma_s_b[(None, stage)],
                tma_bar_ptr=ab_full_mbar + stage,
                mcast_mask=b_full_mask,
            )
            cute.copy(
                tma_sfa,
                tma_g_sfa[(None, k_tile)],
                tma_s_sfa[(None, stage)],
                tma_bar_ptr=ab_full_mbar + stage,
                mcast_mask=sfa_full_mask,
            )
            cute.copy(
                tma_sfb,
                tma_g_sfb[(None, k_tile)],
                tma_s_sfb[(None, stage)],
                tma_bar_ptr=ab_full_mbar + stage,
                mcast_mask=sfb_full_mask,
            )
            if is_iket_cluster:
                cute.experimental.iket.range_pop()  # tma_issue
                cute.experimental.iket.range_pop()  # tma_k_tile

        # Drain every slot.  This keeps both CTAs alive until the last
        # remote tcgen05 commit-arrive has reached their empty barriers.
        if is_iket_cluster:
            cute.experimental.iket.range_push("tma_tail")
        for tail_offset in range(AB_STAGES):
            linear_stage = K_TILES + tail_offset
            stage = linear_stage % AB_STAGES
            empty_phase = 1 ^ ((linear_stage // AB_STAGES) % 2)
            cute.arch.mbarrier_wait(
                ab_empty_mbar + stage, cutlass.Int32(empty_phase)
            )
        if is_iket_cluster:
            cute.experimental.iket.range_pop()  # tma_tail
            cute.experimental.iket.range_pop()  # tma_main

    # MMA consumer warp: only the pair leader issues tcgen05 operations.  Its
    # commit-arrive releases each SMEM slot only after the asynchronous MMA has
    # finished reading it.
    if warp == MMA_WARP:
        if is_iket_cluster:
            cute.experimental.iket.range_push("mma_main")
            cute.experimental.iket.range_push("mma_tmem_wait")
        tmem.wait_for_alloc()
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
        acc_ptr = tmem.retrieve_ptr(F32)
        acc = cute.make_tensor(acc_ptr, fake_acc.layout)

        if is_leader_cta:
            if is_iket_cluster:
                cute.experimental.iket.range_push("mma_wait_acc_empty")
            cute.arch.mbarrier_wait(acc_empty_mbar, cutlass.Int32(1))
            if is_iket_cluster:
                cute.experimental.iket.range_pop()

            sfa_ptr = cute.recast_ptr(
                acc_ptr + tcgen05.find_tmem_tensor_col_offset(acc), dtype=SF8
            )
            t_sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
                mma,
                MMA_TILE,
                SF_VECTOR_SIZE,
                cute.slice_(s_layout_sfa, (None, None, None, 0)),
            )
            t_sfa = cute.make_tensor(sfa_ptr, t_sfa_layout)
            sfb_ptr = cute.recast_ptr(
                acc_ptr
                + tcgen05.find_tmem_tensor_col_offset(acc)
                + tcgen05.find_tmem_tensor_col_offset(t_sfa),
                dtype=SF8,
            )
            t_sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
                mma,
                MMA_TILE,
                SF_VECTOR_SIZE,
                cute.slice_(s_layout_sfb, (None, None, None, 0)),
            )
            t_sfb = cute.make_tensor(sfb_ptr, t_sfb_layout)

            s2t_atom = cute.make_copy_atom(
                tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.TWO), SF8
            )
            s_sfa_compact = cute.filter_zeros(s_sfa)
            t_sfa_compact = cute.filter_zeros(t_sfa)
            copy_sfa = tcgen05.make_s2t_copy(s2t_atom, t_sfa_compact)
            copy_sfa_thr = copy_sfa.get_slice(0)
            copy_sfa_src = tcgen05.get_s2t_smem_desc_tensor(
                copy_sfa, copy_sfa_thr.partition_S(s_sfa_compact)
            )
            copy_sfa_dst = copy_sfa_thr.partition_D(t_sfa_compact)

            s_sfb_compact = cute.filter_zeros(s_sfb)
            t_sfb_compact = cute.filter_zeros(t_sfb)
            copy_sfb = tcgen05.make_s2t_copy(s2t_atom, t_sfb_compact)
            copy_sfb_thr = copy_sfb.get_slice(0)
            copy_sfb_src = tcgen05.get_s2t_smem_desc_tensor(
                copy_sfb, copy_sfb_thr.partition_S(s_sfb_compact)
            )
            copy_sfb_dst = copy_sfb_thr.partition_D(t_sfb_compact)

            mma.set(tcgen05.Field.ACCUMULATE, False)
            for k_tile in cutlass.range(K_TILES, unroll=1):
                if is_iket_cluster:
                    cute.experimental.iket.range_push("mma_k_tile", k_tile)
                stage = k_tile % AB_STAGES
                full_phase = (k_tile // AB_STAGES) % 2
                if is_iket_cluster:
                    cute.experimental.iket.range_push("mma_wait_ab_full")
                cute.arch.mbarrier_wait(ab_full_mbar + stage, full_phase)
                if is_iket_cluster:
                    cute.experimental.iket.range_pop()

                s2t_stage = (None, None, None, None, stage)
                if is_iket_cluster:
                    cute.experimental.iket.range_push("mma_s2t")
                cute.copy(copy_sfa, copy_sfa_src[s2t_stage], copy_sfa_dst)
                cute.copy(copy_sfb, copy_sfb_src[s2t_stage], copy_sfb_dst)
                if is_iket_cluster:
                    cute.experimental.iket.range_pop()

                if is_iket_cluster:
                    cute.experimental.iket.range_push("mma_issue")
                for kblock in cutlass.range(
                    cute.size(r_a, mode=[2]), unroll_full=True
                ):
                    mma.set(
                        tcgen05.Field.SFA,
                        t_sfa[(None, None, kblock)].iterator,
                    )
                    mma.set(
                        tcgen05.Field.SFB,
                        t_sfb[(None, None, kblock)].iterator,
                    )
                    operand_coord = (None, None, kblock, stage)
                    cute.gemm(
                        mma,
                        acc,
                        r_a[operand_coord],
                        r_b[operand_coord],
                        acc,
                    )
                    mma.set(tcgen05.Field.ACCUMULATE, True)
                if is_iket_cluster:
                    cute.experimental.iket.range_pop()  # mma_issue

                if is_iket_cluster:
                    cute.experimental.iket.range_push("mma_release_ab")
                with cute.arch.elect_one():
                    tcgen05.commit(
                        ab_empty_mbar + stage,
                        ab_empty_mask,
                        tcgen05.CtaGroup.TWO,
                    )
                if is_iket_cluster:
                    cute.experimental.iket.range_pop()  # mma_release_ab
                    cute.experimental.iket.range_pop()  # mma_k_tile

            if is_iket_cluster:
                cute.experimental.iket.range_push("mma_commit_acc")
            with cute.arch.elect_one():
                tcgen05.commit(
                    acc_full_mbar,
                    acc_full_mask,
                    tcgen05.CtaGroup.TWO,
                )
            if is_iket_cluster:
                cute.experimental.iket.range_pop()

            # ACC_STAGES=1: after one advance the next producer phase is 0.
            if is_iket_cluster:
                cute.experimental.iket.range_push("mma_wait_acc_empty_tail")
            cute.arch.mbarrier_wait(acc_empty_mbar, cutlass.Int32(0))
            if is_iket_cluster:
                cute.experimental.iket.range_pop()
        if is_iket_cluster:
            cute.experimental.iket.range_pop()  # mma_main

    # Four epilogue warps per CTA drain that CTA's 128x256 accumulator half.
    # One elected lane per warp contributes a remote empty arrival, so the pair
    # leader waits for 2 CTAs * 4 epilogue warps before TMEM can be reused.
    if warp < MMA_WARP:
        if is_iket_cluster:
            cute.experimental.iket.range_push("epi_main")
            cute.experimental.iket.range_push("epi_tmem_wait")
        tmem.wait_for_alloc()
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
        acc_ptr = tmem.retrieve_ptr(F32)
        acc = cute.make_tensor(acc_ptr, fake_acc.layout)

        if is_iket_cluster:
            cute.experimental.iket.range_push("epi_setup")
        t2r_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(
                tcgen05.Repetition.x128, tcgen05.Pack.NONE
            ),
            F32,
        )
        t2r = tcgen05.make_tmem_copy(t2r_atom, acc)
        t2r_thr = t2r.get_slice(tid)
        t_acc = t2r_thr.partition_S(acc)
        t_out = t2r_thr.partition_D(mma_g_c)
        r_acc = cute.make_rmem_tensor(
            t_out[None, None, None, None, 0, 0, 0].shape, F32
        )
        r_out = cute.make_rmem_tensor(
            t_out[None, None, None, None, 0, 0, 0].shape, BF16
        )
        out = t_out[(None, None, None, None, *mma_tile_coord_mnl)]
        if is_iket_cluster:
            cute.experimental.iket.range_pop()  # epi_setup

        if is_iket_cluster:
            cute.experimental.iket.range_push("epi_wait_acc")
        cute.arch.mbarrier_wait(acc_full_mbar, cutlass.Int32(0))
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("epi_t2r")
        cute.copy(t2r, t_acc, r_acc)
        r_out.store(r_acc.load().to(BF16))
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("epi_store")
        cute.copy(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BF16),
            r_out,
            out,
        )
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_push("epi_release_acc")
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(acc_empty_mbar, acc_empty_dst_rank)
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
            cute.experimental.iket.range_pop()  # epi_main

    # The full CTA rendezvous prevents the allocator warp from freeing TMEM
    # while another local role is still using its address or metadata.
    if is_iket_cluster:
        cute.experimental.iket.range_push("cta_tail_sync")
    cute.arch.barrier()
    if is_iket_cluster:
        cute.experimental.iket.range_pop()
    if warp < MMA_WARP:
        if is_iket_cluster:
            cute.experimental.iket.range_push("tmem_dealloc")
        tmem.relinquish_alloc_permit()
        tmem.free(tmem.retrieve_ptr(F32))
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
    if is_iket_cluster:
        cute.experimental.iket.range_pop()  # dense_gemm_v10


@cute.jit
def dense_gemm_v10(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    stream,
):
    """Build and launch the fixed non-persistent 2SM MXFP8 GEMM."""

    a = cute.make_tensor(a_ptr, cute.make_layout((M, K, 1), stride=(K, 1, M * K)))
    b = cute.make_tensor(b_ptr, cute.make_layout((N, K, 1), stride=(K, 1, N * K)))
    c = cute.make_tensor(c_ptr, cute.make_layout((M, N, 1), stride=(N, 1, M * N)))

    # sf_atom = cute.make_layout(
    #      M 维度     K 维度
    #     ((32, 4), (SF_VECTOR_SIZE, 4)),
    #     stride=((16, 4), (0, 1)),
    # )
    # Swizzle32x4x4 是 NVIDIA Blackwell tcgen05.mma 的硬件输入格式要求
    # sfa = cute.make_tensor(
    #     sfa_ptr, cute.tile_to_shape(sf_atom, a.shape, (2, 1, 3))
    # )
    # sfb = cute.make_tensor(
    #     sfb_ptr, cute.tile_to_shape(sf_atom, b.shape, (2, 1, 3))
    # )
    #
    # 当前 M=N=K=16384、SF_VECTOR_SIZE=32 时，sfa.layout 与 sfb.layout 均为：
    #   shape  = (((32, 4), 128), ((32, 4), 128), 1)
    #   stride = (((16, 4), 65536), ((0, 1), 512), 8388608)
    sfa = cute.make_tensor(
        sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VECTOR_SIZE)
    )
    sfb = cute.make_tensor(
        sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VECTOR_SIZE)
    )


    #   mma_op = tcgen05.MmaMXF8F6F4Op(
    #       FP8, FP8, (256, 256, 32), CtaGroup.TWO,
    #       OperandSource.SMEM, OperandMajorMode.K, OperandMajorMode.K,
    #   )
    #   mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op))
    # - 两个 K-major 指定 A 的 (M,K) 与 B 的 (N,K) 都让 K 成为连续主维，
    #   并据此生成后续 MMA partition 和 SMEM descriptor/layout。
    # - SF8 + SF_VECTOR_SIZE=32 表示 SFA/SFB 是 UE8M0，连续 32 个 K 元素
    #   共享一个 scale；这正是当前 MXFP8 指令固定支持的 scale contract。
    # - CtaGroup.TWO 让一个 MMA atom 由相邻两个 CTA/SM 协作，得到
    #   mma.thr_id = 2:1；
    #   256 的 M 均分后每个 CTA 负责 128x256 的 C 子块。
    # - MMA_TILE[:2]=(256,256) 只传 M/N。helper 为 MXFP8 补上固定 K=32，
    #   所以一个 MMA_TILE=(256,256,128) 的 K-stage 后续要发 128/32=4 次 MMA。
    #
    # mma.shape_mnk      = (256, 256, 32)
    # mma.thr_id            # 2:1
    # MMA_TILE           = (256, 256, 128)  # 一个流水 stage
    mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
        FP8,
        FP8,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF8,
        SF_VECTOR_SIZE,
        tcgen05.CtaGroup.TWO,
        MMA_TILE[:2],
    )
    # mma 是真正发 MMA 指令的 atom，mma_sfb 只是一个"布局工具"，专门用来构造
    # SFB 的 TMA atom 和 partition。
    # - A：128(M) × 128(K) = 16 KiB，按 M 切。
    # - B：128(N) × 128(K) = 16 KiB，按 N 切。
    # - SFA：128(M) × 4 = 512 B，跟着 M 切。
    # - SFB：1 KiB = 256(N) × 4，也就是完整的 N=256，没有切。
    # SFB 之所以不切，是因为 scale factor 不像 A/B 那样被 MMA 直接从对端
    # SMEM 读走，而要先在本 CTA 内做 SMEM→TMEM 拷贝。
    mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
        FP8,
        FP8,
        OperandMajorMode.K,
        OperandMajorMode.K,
        SF8,
        SF_VECTOR_SIZE,
        tcgen05.CtaGroup.ONE,
        SFB_MMA_TILE[:2],
    )

    # cute.tiled_divide 是 CuTe 的 Layout 除法/分块操作
    # 把原 CTA 坐标重新表达成“atom 内坐标 + atom 外坐标”
    # tiled_divide 的结果形式是：
    # (Tiler, RestM, RestN, RestK, ...)
    #
    # CLUSTER_SHAPE (2, 1, 1)
    #
    # mma.thr_id            # 2:1
    # mma.thr_id.shape      # 2
    #
    # mma_sfb.thr_id        # 1:0
    # mma_sfb.thr_id.shape  # 1
    #
    # cluster_layout_vmnk      # ((2),1,1,1):((1),0,0,0)
    # cluster_layout_sfb_vmnk  # ((1),2,1,1):((0),1,0,0)
    cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE), (mma.thr_id.shape,)
    )
    cluster_layout_sfb_vmnk = cute.tiled_divide(
        cute.make_layout(CLUSTER_SHAPE), (mma_sfb.thr_id.shape,)
    )
    # make_smem_layout_a
    # - a_mk_tile = cute.dice(MMA_TILE, (1, None, 1)) # (M,N,K) 中删除与 A 无关的 N
    #   (256, 128)
    #   # 按 tiled MMA 的 2CTA ownership 和指令 K=32 分解 A tile。
    # - a_smem_shape = mma.partition_shape_A(a_mk_tile)
    #   ((128, 32), 1, 4)
    #    - (128, 32) 每 CTA 每条指令的 (M, K)
    #    - 1，M 方向几条指令
    #    - 4, K 方向几条指令
    # - a_smem_shape_m_k
    #   (128, 128)
    #   a_smem_layout_atom = tcgen05.make_smem_layout_atom(
    #       SmemLayoutAtomKind.K_SW128,
    #       FP8,
    #   )
    #   S<3,4,3> o 0 o (8,128):(128,1)
    # - a_smem_layout_atom
    # - a_smem_shape = cute.append(a_smem_shape, AB_STAGES)
    #   ((128,32), 1, 4, 4)
    # - s_layout_a = tcgen05.tile_to_mma_shape(
    #       a_smem_layout_atom,
    #       a_smem_shape,
    #       order=order,
    #   )
    # S<3,4,3> o 0 o
    # ((128,32),1,4,4):((128,1),0,32,16384)
    s_layout_a = sm100_utils.make_smem_layout_a(mma, MMA_TILE, FP8, AB_STAGES)
    s_layout_b = sm100_utils.make_smem_layout_b(mma, MMA_TILE, FP8, AB_STAGES)
    # s_layout_sfa =
    # ((((32,4),1),(32,1)),1,4,4):
    # ((((16,4),0),(0,0)),0,1,512)
    #
    # s_layout_sfb =
    # ((((32,4),2),(32,1)),1,4,4):
    # ((((16,4),512),(0,0)),0,1,1024)
    s_layout_sfa = blockscaled_utils.make_smem_layout_sfa(
        mma, MMA_TILE, SF_VECTOR_SIZE, AB_STAGES
    )
    s_layout_sfb = blockscaled_utils.make_smem_layout_sfb(
        mma, MMA_TILE, SF_VECTOR_SIZE, AB_STAGES
    )

    # 根据 cluster shape 选择哪种 TMA 指令，并做合法性检查
    # 2-CTA 普通 G2S
    a_op = sm100_utils.cluster_shape_to_tma_atom_A(
        CLUSTER_SHAPE[:2], mma.thr_id
    )
    b_op = sm100_utils.cluster_shape_to_tma_atom_B(
        CLUSTER_SHAPE[:2], mma.thr_id
    )
    sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
        CLUSTER_SHAPE[:2], mma.thr_id
    )
    # 2-CTA multicast G2S
    sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
        CLUSTER_SHAPE[:2], mma.thr_id
    )

    # 根据 GMEM layout、SMEM layout、MMA tile 和 CTA ownership 构造 TMA CopyAtom 与坐标 Tensor
    tma_a, g_a = cute.nvgpu.make_tiled_tma_atom_A(
        a_op,
        a,
        cute.slice_(s_layout_a, (None, None, None, 0)),
        MMA_TILE,
        mma,
        cluster_layout_vmnk.shape,
    )
    tma_b, g_b = cute.nvgpu.make_tiled_tma_atom_B(
        b_op,
        b,
        cute.slice_(s_layout_b, (None, None, None, 0)),
        MMA_TILE,
        mma,
        cluster_layout_vmnk.shape,
    )
    tma_sfa, g_sfa = cute.nvgpu.make_tiled_tma_atom_A(
        sfa_op,
        sfa,
        cute.slice_(s_layout_sfa, (None, None, None, 0)),
        MMA_TILE,
        mma,
        cluster_layout_vmnk.shape,
        internal_type=cutlass.Int16,
    )
    tma_sfb, g_sfb = cute.nvgpu.make_tiled_tma_atom_B(
        sfb_op,
        sfb,
        cute.slice_(s_layout_sfb, (None, None, None, 0)),
        SFB_MMA_TILE,
        mma_sfb,
        cluster_layout_sfb_vmnk.shape,
        internal_type=cutlass.Int16,
    )

    # The leader's full barrier accounts for transactions contributed by both
    # CTAs in the two-CTA MMA atom.
    tma_bytes = (
        cute.size_in_bytes(FP8, cute.slice_(s_layout_a, (None, None, None, 0)))
        + cute.size_in_bytes(FP8, cute.slice_(s_layout_b, (None, None, None, 0)))
        + cute.size_in_bytes(SF8, cute.slice_(s_layout_sfa, (None, None, None, 0)))
        + cute.size_in_bytes(SF8, cute.slice_(s_layout_sfb, (None, None, None, 0)))
    ) * cute.size(mma.thr_id.shape)

    _dense_gemm_v10_kernel(
        mma,
        mma_sfb,
        tma_a,
        g_a,
        tma_b,
        g_b,
        tma_sfa,
        g_sfa,
        tma_sfb,
        g_sfb,
        c,
        cluster_layout_vmnk,
        cluster_layout_sfb_vmnk,
        s_layout_a,
        s_layout_b,
        s_layout_sfa,
        s_layout_sfb,
        tma_bytes,
    ).launch(
        # Regular grid: every CTA executes exactly one output half-tile.
        grid=(M // CTA_TILE[0], N // CTA_TILE[1], 1),
        block=(THREADS, 1, 1),
        cluster=CLUSTER_SHAPE,
        stream=stream,
    )


__all__ = [
    "AB_STAGES",
    "ACC_STAGES",
    "BF16",
    "CLUSTER_SHAPE",
    "CTA_TILE",
    "EPILOGUE_WARPS",
    "F32",
    "FP8",
    "MMA_INSTRUCTION",
    "MMA_TILE",
    "MMA_WARP",
    "SFB_MMA_TILE",
    "SF8",
    "SF_VECTOR_SIZE",
    "THREADS",
    "TMA_WARP",
    "dense_gemm_v10",
]
