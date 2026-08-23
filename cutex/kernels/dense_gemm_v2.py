"""固定 B300 精度合同下的单级 TMA + CUDA Core MXFP8 GEMM 教学实现。"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync

from .dense_gemm_contract import K, M, N, SF_VECTOR_SIZE


FP8 = cutlass.Float8E4M3FN
SF8 = cutlass.Float8E8M0FNU
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

# 每个 thread 负责一个输出。A/B 每轮搬运 16x128；scale 则按其原生
# 128x128 Swizzle32x4x4 物理块搬运，避免手工解释 TE 的 swizzle 地址。
TILE = (16, 16, 128)  # M, N, K
SCALE_TILE = (128, 128)  # outer, K
THREADS, TMA_STAGES = 256, 1
SCALE_CTAS_PER_TILE = SCALE_TILE[0] // TILE[0]


@cute.struct
class SharedStorage:
    # 一个 64-bit mbarrier，反复用 phase=0/1 表示相邻两轮 TMA 完成事件。
    tma_mbar: cute.struct.MemRange[cutlass.Int64, 1]

# TODO: https://zhuanlan.zhihu.com/p/2007400131314595305

@cute.kernel
def _dense_gemm_v2_kernel(
    tma_a: cute.CopyAtom,
    g_a: cute.Tensor,
    tma_b: cute.CopyAtom,
    g_b: cute.Tensor,
    tma_sfa: cute.CopyAtom,
    g_sfa: cute.Tensor,
    tma_sfb: cute.CopyAtom,
    g_sfb: cute.Tensor,
    g_c: cute.Tensor,
    s_layout_a: cute.Layout,
    s_layout_b: cute.Layout,
    s_layout_sfa: cute.Layout,
    s_layout_sfb: cute.Layout,
    tma_bytes: cutlass.Constexpr,
):
    """Load each K tile with TMA, wait, then accumulate on CUDA Cores."""

    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tid, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, bid_l = cute.arch.block_idx()
    local_m, local_n = tid // TILE[1], tid % TILE[1]

    # smem 分配，构造 tensor
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    s_a = smem.allocate_tensor(FP8, s_layout_a, byte_alignment=128)
    s_b = smem.allocate_tensor(FP8, s_layout_b, byte_alignment=128)
    s_sfa = smem.allocate_tensor(SF8, s_layout_sfa, byte_alignment=128)
    s_sfb = smem.allocate_tensor(SF8, s_layout_sfb, byte_alignment=128)

    # local_tile 只描述逻辑分块，不搬数据：
    #   l_a/l_b 保留当前 CTA 的全部 K tiles；
    #   l_sfa/l_sfb 按 scale 的原生 128x128 存储块分块。
    # 后面的 tma_partition 再把这些 GMEM tile 与对应 SMEM tile 配对。
    l_a = cute.local_tile(
        g_a, cute.slice_(TILE, (None, 0, None)), (None, None, None)
    )
    l_b = cute.local_tile(
        g_b, cute.slice_(TILE, (0, None, None)), (None, None, None)
    )
    l_sfa = cute.local_tile(g_sfa, SCALE_TILE, (None, None, None))
    l_sfb = cute.local_tile(g_sfb, SCALE_TILE, (None, None, None))

    # tma_partition 把 make_tiled_tma_atom 生成的 TMA atom 作用到具体 tensor：
    #   输入  = CopyAtom + CTA rank/layout + SMEM view + GMEM view
    #   输出  = tma_s_*（TMA 目标 SMEM view）和 tma_g_*（TMA 源坐标 view）
    # group_modes(..., 0, 2) 把 tile 的两个逻辑维度合成 TMA atom mode，
    # 剩余 modes（CTA M/N、K tile、batch）保留下来供后续索引。
    cta_layout = cute.make_layout(1)
    tma_s_a, tma_g_a = cpasync.tma_partition(
        tma_a,
        0,
        cta_layout,
        cute.group_modes(s_a, 0, 2),
        cute.group_modes(l_a, 0, 2),
    )
    tma_s_b, tma_g_b = cpasync.tma_partition(
        tma_b,
        0,
        cta_layout,
        cute.group_modes(s_b, 0, 2),
        cute.group_modes(l_b, 0, 2),
    )
    tma_s_sfa, tma_g_sfa = cpasync.tma_partition(
        tma_sfa,
        0,
        cta_layout,
        cute.group_modes(s_sfa, 0, 2),
        cute.group_modes(l_sfa, 0, 2),
    )
    tma_s_sfb, tma_g_sfb = cpasync.tma_partition(
        tma_sfb,
        0,
        cta_layout,
        cute.group_modes(s_sfb, 0, 2),
        cute.group_modes(l_sfb, 0, 2),
    )
    # scale layout 在同一个 32-value 向量内使用 zero stride 广播同一 scale。
    # TMA 实际只搬唯一物理元素，因此去掉 zero-stride modes。
    # tma_s_sfa = cute.filter_zeros(tma_s_sfa)
    # tma_g_sfa = cute.filter_zeros(tma_g_sfa)
    tma_s_sfb = cute.filter_zeros(tma_s_sfb)
    tma_g_sfb = cute.filter_zeros(tma_g_sfb)

    # 一个 16-row CTA 只消费 128-row scale 块中的 1/8；整个原生块仍由 TMA
    # 一次搬入 SMEM，scale_row_* 再选择本 CTA 对应的 16 行。
    scale_tile_m = bid_m // SCALE_CTAS_PER_TILE
    scale_tile_n = bid_n // SCALE_CTAS_PER_TILE
    scale_row_a = (bid_m % SCALE_CTAS_PER_TILE) * TILE[0] + local_m
    scale_row_b = (bid_n % SCALE_CTAS_PER_TILE) * TILE[1] + local_n

    tma_g_a = tma_g_a[(None, bid_m, None, bid_l)]
    tma_g_b = tma_g_b[(None, bid_n, None, bid_l)]
    tma_g_sfa = tma_g_sfa[(None, scale_tile_m, None, bid_l)]
    tma_g_sfb = tma_g_sfb[(None, scale_tile_n, None, bid_l)]

    tma_mbar = storage.tma_mbar.data_ptr()
    if warp == 0:
        with cute.arch.elect_one():
            # 1 代表一个参与者，一次 arrive 就行
            cute.arch.mbarrier_init(tma_mbar, 1)
    # 发布前面的 mbarrier 初始化
    cute.arch.mbarrier_init_fence()
    # 保证所有 thread 看见 barrier 初始化。
    cute.arch.sync_threads()

    phase = cutlass.Int32(0)
    acc = F32(0.0)
    k_tiles = cute.size(l_a, mode=[3])
    for k_tile in cutlass.range(k_tiles, unroll=1):
        if warp == 0:
            # 声明本 phase 要等待多少 TMA transaction bytes，
            # 同时完成唯一一次 producer arrival。
            # pending_arrival: 1 -> 0    
            # tx_count:        0 -> tma_bytes
            with cute.arch.elect_one():
                # 一般都是先 expect 再 arrive，arrive 变成 0 之前(barrier 未完成之前)都能 expect
                cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_bytes)

            # cute.copy 自己会在 warp 内选择一个 issuing thread，
            # 所以不能把 cute.copy 放进 elect_one()。
            cute.copy(
                # copy atom
                tma_a,
                # tma gmem view
                tma_g_a[(None, k_tile)],
                # tma smem view
                tma_s_a[(None, 0)],
                # barrier
                tma_bar_ptr=tma_mbar,
            )
            cute.copy(
                tma_b,
                tma_g_b[(None, k_tile)],
                tma_s_b[(None, 0)],
                tma_bar_ptr=tma_mbar,
            )
            cute.copy(
                tma_sfa,
                tma_g_sfa[(None, k_tile)],
                tma_s_sfa[(None, 0)],
                tma_bar_ptr=tma_mbar,
            )
            cute.copy(
                tma_sfb,
                tma_g_sfb[(None, k_tile)],
                tma_s_sfb[(None, 0)],
                tma_bar_ptr=tma_mbar,
            )

        # 所有 thread 都等待 arrival==0 且四路 TMA transaction 全部完成；
        # wait 返回后才允许读取 s_a/s_b/s_sfa/s_sfb。
        # phase 表示代际，否则无法区分多代之间的状态
        cute.arch.mbarrier_wait(tma_mbar, phase)
        phase = phase ^ cutlass.Int32(1)

        for kk in cutlass.range(TILE[2], unroll=1):
            a = s_a[(local_m, kk, 0)].to(F32)
            b = s_b[(local_n, kk, 0)].to(F32)
            sa = s_sfa[(scale_row_a, kk, 0)].to(F32)
            sb = s_sfb[(scale_row_b, kk, 0)].to(F32)
            acc = acc + (a * sa) * (b * sb)

        # 这里只用一个 SMEM stage。轮末 CTA 同步保证所有 warp 已读完当前 tile，
        # 否则 warp 0 可能过早发出下一轮 TMA 并覆盖仍在被读取的 SMEM。
        cute.arch.sync_threads()

    global_m = bid_m * TILE[0] + local_m
    global_n = bid_n * TILE[1] + local_n
    g_c[(global_m, global_n, bid_l)] = acc.to(BF16)


@cute.jit
def dense_gemm_v2(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    stream,
):
    """Launch ``BF16 Y = MXFP8 X @ W.T`` with TMA and scalar FP32 FMA."""
    a = cute.make_tensor(a_ptr, cute.make_layout((M, K, 1), stride=(K, 1, M * K)))
    b = cute.make_tensor(b_ptr, cute.make_layout((N, K, 1), stride=(K, 1, N * K)))
    c = cute.make_tensor(c_ptr, cute.make_layout((M, N, 1), stride=(N, 1, M * N)))
    sfa = cute.make_tensor(
        sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VECTOR_SIZE)
    )
    sfb = cute.make_tensor(
        sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VECTOR_SIZE)
    )

    # 定义 smem 上的 layout
    s_layout_a = cute.make_layout(
        (TILE[0], TILE[2], TMA_STAGES),
        stride=(TILE[2], 1, TILE[0] * TILE[2]),
    )
    s_layout_b = cute.make_layout(
        (TILE[1], TILE[2], TMA_STAGES),
        stride=(TILE[2], 1, TILE[1] * TILE[2]),
    )
    s_layout_sfa = blockscaled_utils.make_smem_layout_sf(
        SCALE_TILE, SF_VECTOR_SIZE, TMA_STAGES
    )
    s_layout_sfb = blockscaled_utils.make_smem_layout_sf(
        SCALE_TILE, SF_VECTOR_SIZE, TMA_STAGES
    )

    # TMA 构造关系：
    #   TMA Op + GMEM tensor + 单 stage SMEM layout + CTA tiler
    #       -> TMA CopyAtom + TMA coordinate tensor
    #
    # tma_op 描述要使用的硬件搬运指令
    tma_op = cpasync.CopyBulkTensorTileG2SOp()

    # make_tiled_tma_atom 将抽象 op 绑定到 A 的具体搬运问题：
    #   a                                  -- A 的逻辑 GMEM tensor；
    #   slice_(s_layout_a, ..., 0)         -- 一个 pipeline stage 的 SMEM 布局；
    #   (TILE_M, TILE_K)                   -- 一个 CTA 搬运的逻辑 tile。
    # 返回
    #   tma_a 是之后交给 tma_partition/cute.copy 的 CopyAtom
    #   g_a 是把 A 的逻辑坐标映射成 TMA descriptor 坐标的 tensor
    tma_a, g_a = cpasync.make_tiled_tma_atom(
        tma_op,
        a,
        # 取出第 0 个 stage 的二维 layout
        # -> 
        # shape  = (TILE_M, TILE_K)
        # stride = (TILE_K, 1)
        cute.slice_(s_layout_a, (None, None, 0)),
        (TILE[0], TILE[2]),
    )
    tma_b, g_b = cpasync.make_tiled_tma_atom(
        tma_op,
        b,
        cute.slice_(s_layout_b, (None, None, 0)),
        (TILE[1], TILE[2]),
    )
    tma_sfa, g_sfa = cpasync.make_tiled_tma_atom(
        tma_op,
        sfa,
        cute.slice_(s_layout_sfa, (None, None, 0)),
        SCALE_TILE,
        internal_type=cutlass.Int16,
    )
    tma_sfb, g_sfb = cpasync.make_tiled_tma_atom(
        tma_op,
        sfb,
        cute.slice_(s_layout_sfb, (None, None, 0)),
        SCALE_TILE,
        internal_type=cutlass.Int16,
    )
    # 给 TMA barrier/pipeline 使用的“本 stage 预计完成多少字节”
    tma_bytes = (
        cute.size_in_bytes(FP8, cute.slice_(s_layout_a, (None, None, 0)))
        + cute.size_in_bytes(FP8, cute.slice_(s_layout_b, (None, None, 0)))
        + cute.size_in_bytes(SF8, cute.slice_(s_layout_sfa, (None, None, 0)))
        + cute.size_in_bytes(SF8, cute.slice_(s_layout_sfb, (None, None, 0)))
    )

    _dense_gemm_v2_kernel(
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
        grid=(M // TILE[0], N // TILE[1], 1),
        block=(THREADS, 1, 1),
        stream=stream,
    )


__all__ = [
    "BF16",
    "F32",
    "FP8",
    "SCALE_TILE",
    "SF8",
    "SF_VECTOR_SIZE",
    "THREADS",
    "TILE",
    "TMA_STAGES",
    "dense_gemm_v2",
]
