"""CLC-persistent 2SM warp-specialized MXFP8 GEMM for fixed 16384 cubed.

The full/empty protocols are expanded instead of hidden behind the generic
pipeline wrappers.  One scheduler warp fetches work through Cluster Launch
Control (CLC), one TMA warp fills a six-stage A/B/SFA/SFB ring, one warp issues
2CTA block-scaled MMA, and four warps drain the BF16 epilogue.  Barrier setup,
TMEM allocation, and the A/B ring tail occur once per persistent CTA pair, not
once per output tile; required input loads and output stores remain.

Fixed-resource accounting per CTA::

    one K=128 stage = 16 KiB A + 16 KiB B + 512 B SFA + 1 KiB SFB
                    = 34,304 B
    six-stage ring  = 205,824 B
    aligned storage = 256 B (manual/CLC barriers and TMEM metadata)
    launch SMEM     = 206,080 B

The 512-column TMEM allocation contains accumulator views ``[0,256)`` and
``[208,464)`` plus SFA ``[464,480)`` and SFB ``[480,512)``.  The views overlap
by 48 columns.  Each epilogue therefore reads that overlap first, executes
``fence_view_async_tmem_load()``, and releases the shared empty barrier before
conversion and stores finish.  Separate full barriers track the two in-flight
accumulator completions without changing the single overlap dependency.

For ``M=N=K=16384``, the CLC launch grid is ``(128,64,1)`` CTAs: 4,096 logical
two-CTA output clusters.  On the measured B300, at most 74 two-CTA clusters are
resident; CLC cancels queued clusters and lets that resident cohort repeatedly
claim work in M-major order with an eight-cluster swizzle.  Every output tile
runs 128 K-stages and four K=32 MMA instructions per stage.

Primary design references are NVIDIA CUTLASS 4.7.0
``dense_blockscaled_gemm_persistent.py`` and
``dense_gemm_persistent_dynamic.py``.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.nvgpu.common import CacheEvictionPriority
from cutlass.pipeline import NamedBarrier

from .dense_gemm_contract import K, M, N, SF_VECTOR_SIZE


FP8 = cutlass.Float8E4M3FN
SF8 = cutlass.Float8E8M0FNU
F32 = cutlass.Float32
BF16 = cutlass.BFloat16

# Pair-wide MMA tile.  CtaGroup.TWO splits its M mode evenly, so one CTA owns
# a 128x256 half tile while the pair cooperatively issues 256x256x32
# block-scaled MMA instructions.  A K=128 stage contains four instructions.
MMA_INSTRUCTION = (256, 256, 32)
# K=128 preserves K_SW128 operand layouts while grouping four K=32 MMA
# instructions and their 32x4x4 canonical scale-factor blocks per TMA stage.
MMA_TILE = (256, 256, 128)
CTA_TILE = (128, 256, 128)

# SFB cannot use the two-CTA SF partition directly in CUTLASS 4.7.0.  Follow
# the official block-scaled examples and build a companion one-CTA tiled MMA
# solely for SFB partition/TMA construction.
SFB_MMA_TILE = (128, 256, 128)

CLUSTER_SHAPE = (2, 1, 1)
# Per CTA, six stages use 205,824 B for tensor payloads.  The barrier struct is
# rounded to the next 128-B tensor boundary, making the launch total 206,080 B.
AB_STAGES, ACC_STAGES = 6, 1
ACC_FULL_STAGES = 2

# CLC dynamically hands out the fixed 64x64 cluster-tile problem.  An
# eight-cluster swizzle with M-major rasterization was the best measured
# locality setting on B300.
CLUSTER_SWIZZLE_SIZE = 8
RASTER_ALONG_M = True
CLUSTER_TILES_M = M // MMA_TILE[0]
CLUSTER_TILES_N = N // MMA_TILE[1]
OUTPUT_CLUSTER_TILES = CLUSTER_TILES_M * CLUSTER_TILES_N

# Four warps drain each CTA's half accumulator.  Separate producer and consumer
# warps are what allow TMA and tcgen05.mma to advance different ring stages.
EPILOGUE_WARPS = (0, 1, 2, 3)
MMA_WARP = 4
TMA_WARP = 5
SCHED_WARP = 6
THREADS = 7 * 32
K_TILES = K // MMA_TILE[2]
TMEM_COLUMNS = 512
MMA_KBLOCKS_PER_STAGE = MMA_TILE[2] // MMA_INSTRUCTION[2]
SFA_TMEM_COLUMNS = (CTA_TILE[0] // 32) * MMA_KBLOCKS_PER_STAGE
SFB_TMEM_COLUMNS = (CTA_TILE[1] // 32) * MMA_KBLOCKS_PER_STAGE
SF_TMEM_COLUMNS = SFA_TMEM_COLUMNS + SFB_TMEM_COLUMNS
ACC_PHYSICAL_STAGES = 2
ACC_STAGE_STRIDE = MMA_TILE[1] - SF_TMEM_COLUMNS
ACC_TMEM_COLUMNS = MMA_TILE[1] * ACC_PHYSICAL_STAGES - SF_TMEM_COLUMNS
# Emit IKET ranges only from logical cluster (0, 0, 0), including both CTAs.
IKET_PERSISTENT_CLUSTER = 0
# Production benchmarking does not need runtime range predicates.  Keeping the
# source markers behind a constexpr lets the compiler erase every associated
# branch from the hot persistent loops.
ENABLE_IKET_RANGES = False


@cute.struct
class SharedStorage:
    """Manual full/empty mbarriers plus TMEM allocator metadata."""

    # First STAGES entries are full barriers; the next STAGES are empty.
    ab_mbar: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar: cute.struct.MemRange[
        cutlass.Int64, ACC_FULL_STAGES + ACC_STAGES
    ]
    tmem_dealloc_mbar: cutlass.Int64
    tmem_holding_buf: cutlass.Int32
    clc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
    clc_response: cute.struct.Align[
        cute.struct.MemRange[cutlass.Int32, 4], 16
    ]


@cute.kernel
def _dense_gemm_v13_kernel(
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
    tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    s_layout_a: cute.ComposedLayout,
    s_layout_b: cute.ComposedLayout,
    s_layout_sfa: cute.Layout,
    s_layout_sfb: cute.Layout,
    epi_tile: cute.Tile,
    tma_bytes: cutlass.Constexpr,
):
    """Walk output tiles persistently with a hand-written async schedule."""

    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    tid, _, _ = cute.arch.thread_idx()
    bid_m, bid_n, bid_l = cute.arch.block_idx()
    is_iket_cluster = (
        cutlass.const_expr(ENABLE_IKET_RANGES)
        and bid_m == 0
        and bid_n == 0
        and bid_l == IKET_PERSISTENT_CLUSTER
    )
    if is_iket_cluster:
        cute.experimental.iket.range_push("dense_gemm_v13")
        cute.experimental.iket.range_push("v13_prologue")

    if warp == TMA_WARP:
        cpasync.prefetch_descriptor(tma_a)
        cpasync.prefetch_descriptor(tma_b)
        cpasync.prefetch_descriptor(tma_sfa)
        cpasync.prefetch_descriptor(tma_sfb)

    cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
    is_first_cta_in_cluster = cta_rank == 0
    # cluster_layout_vmnk     = ((2),1,1,1):((1),0,0,0)
    # cluster_layout_sfb_vmnk = ((1),2,1,1):((0),1,0,0)
    # get_flat_coord(rank) 同样是逆映射：
    #   2CTA 视角：rank 0 -> (v=0,...), rank 1 -> (v=1,...)  两个 CTA 是
    #     同一 MMA atom 内的两个 "值线程"。
    #   SFB 1CTA 视角：rank 0 -> (0,m=0,...), rank 1 -> (0,m=1,...)  两个
    #     CTA 变成 M 方向两个独立 tile。
    # 同一个 cta_rank 经两种 layout 得到两套坐标，后面 A/B 用 vmnk 坐标做
    # 半分 partition，SFB 用 sfb_vmnk 坐标做整份 partition。
    cta_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank)
    cta_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank)

    mma_tile_coord_v = bid_m % cute.size(cluster_layout_vmnk, mode=[0])
    is_leader_cta = mma_tile_coord_v == 0

    # AB_STAGES independent SMEM slots form the A/B/SFA/SFB ring.
    #
    # allocate_tensor 把 ComposedLayout 拆成两半：
    #   outer = ((128,32),1,4,6):((128,1),0,32,16384)   -> 决定分配大小和寻址
    #   inner = S<3,4,3>                                 -> 挂在指针上的地址变换
    # 之后任何 s_a[coord] 的地址计算都是 swizzle(outer(coord))：
    #   swizzle(addr) = addr XOR (((addr >> 7) & 0b111) << 4)
    # 即行号（bit 7-9）异或到 16B 列块（bit 4-6）上。CuTe 只负责把这个函数
    # 组合进指针；真正消费它的是 TMA descriptor 和 MMA SMEM descriptor 里的
    # "128B swizzle" 枚举位，二者必须与 S<3,4,3> 一致。
    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)

    # CLC keeps one cancellation response in a single-stage pipeline.  CTA 0's
    # scheduler warp is the sole producer.  One elected lane in each role warp
    # consumes the response, so the empty barrier counts 13 role warps rather
    # than all 448 threads in the pair.
    clc_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    num_clc_consumers = (
        1
        + cute.size(CLUSTER_SHAPE)
        * (1 + len(EPILOGUE_WARPS) + 1)
    )
    clc_consumer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, num_clc_consumers
    )
    clc_pipeline = pipeline.PipelineClcFetchAsync.create(
        barrier_storage=storage.clc_mbar.data_ptr(),
        num_stages=1,
        producer_group=clc_producer_group,
        consumer_group=clc_consumer_group,
        tx_count=16,
        cta_layout_vmnk=cluster_layout_vmnk,
        defer_sync=True,
    )
    clc_response_ptr = storage.clc_response.data_ptr()
    clc_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, 1
    )

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
    # SMEM 上的 SFA / SFB。layout 由 host 侧 make_smem_layout_sfa/sfb 构造
    # （见下面那一节注释，含 atom/stride），这里给地址代数和字节图。
    # 两处共用同一 canonical 排布，SFB=N=256，SFA=M=128。
    #
    # SFA 单 stage 的紧凑寻址（滤掉 stride=0 的广播模式后）：
    #   byte(m, kb) = (m % 32) * 16 + (m // 32) * 4 + kb        kb = 0..3
    # 即位域  [8:4]=m_lo(0..31)  [3:2]=m_hi(0..3)  [1:0]=kb(0..3)，
    # 值域恰好 0..511，128 行 x 4 kblock 双射到 512 B。
    #
    # 字节图（一行 = 16 B，正好是 tcgen05.cp 32x128b 一个 lane 的份额）：
    #
    #   m_lo  byte0..3       byte4..7       byte8..11      byte12..15
    #   ----  -------------  -------------  -------------  -------------
    #    0    m=0  kb=0..3   m=32 kb=0..3   m=64 kb=0..3   m=96 kb=0..3
    #    1    m=1  kb=0..3   m=33 kb=0..3   m=65 kb=0..3   m=97 kb=0..3
    #   ...
    #    31   m=31 kb=0..3   m=63 kb=0..3   m=95 kb=0..3   m=127 kb=0..3
    #
    # 这就是 "32x4x4" canonical 排布：32 行(m_lo) x 4 行组(m_hi) x 4 K-block，
    # 每字节一个 UE8M0 scale（32 个 K 元素共享 1 个）。一个 scale 覆盖
    # K 范围 kb*32..kb*32+31，对应主循环里 kblock 切 t_sfa[(None,None,kblock)]
    # 的那 4 条 MMA。
    #
    # SFB 仅是上表整体两倍：N=256 拆成两个 128 半块，第二半块从 byte 512
    # 开始（layout shape 里那个 2、stride 512），故每 stage SFA=512 B、
    # SFB=1 KiB，跨 stage 步长分别 512/1024 B。
    #
    # SF 没有 swizzle（不含 S<B,M,S>），tcgen05 对 SF 的 SMEM descriptor
    # 只认这种 canonical 排布。SF 与 TMEM 的对应见 kernel 里 t_sfa/t_sfb
    # 那一节注释。

    # Manual barrier layout.  Each slot has a full generation and an empty
    # generation.  Producer phases start at 1 (an initialized empty barrier is
    # immediately reusable); consumer phases start at 0.
    ab_full_mbar = storage.ab_mbar.data_ptr()
    ab_empty_mbar = ab_full_mbar + AB_STAGES
    acc_full_mbar = storage.acc_mbar.data_ptr()
    acc_empty_mbar = acc_full_mbar + ACC_FULL_STAGES

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
            for stage in range(ACC_FULL_STAGES):
                cute.arch.mbarrier_init(acc_full_mbar + stage, 1)
            for stage in range(ACC_STAGES):
                cute.arch.mbarrier_init(
                    acc_empty_mbar + stage, num_acc_empty_arrivals
                )

    # TMEM allocation is synchronized only across the MMA warp and four
    # epilogue warps.  The TMA warp never touches TMEM.
    tmem_alloc_barrier = NamedBarrier(
        barrier_id=2,
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
    #
    # local_tile = zipped_divide + slice，即 Layout 除法：
    #   g_a : (16384, 16384, 1)，tiler = slice_(MMA_TILE,(None,0,None)) = (256,128)
    #   除法把每个模式拆成 (tile 内, tile 外)：
    #     16384/256 -> (256, 64)   16384/128 -> (128, 128)
    #   保留 (None,None,None) 的 tile-外坐标后：
    #     l_a : ((256,128), 64, 128, 1)  = (tile 形状, m 号, k 号, l 号)
    # 除法只重排 stride，不搬数据；此后 "第 (m,k) 个 tile" 变成一个普通下标。
    # l_b 的 tiler 是 (256,128)（N,K），同样得 ((256,128), 64, 128, 1)；
    # l_c 的 tiler 是 (256,256)，得 ((256,256), 64, 64, 1)。
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

    # partition_A/B/C 是 "composition with the MMA thr/val layout"：
    # get_slice(v) 固定 atom 内的值线程 v（即本 CTA 在 pair 里的角色），
    # 然后把 tile 内坐标 (256,...) 重写成 (本 CTA 的 (128,K), 指令重复数)。
    #   mma_g_a : ((128,32), 1, 4, 64, 128, 1)
    #             = (指令内 (M,K), M 重复, K 重复, tile_m, tile_k, l)
    # v=0 取 M 行 0-127，v=1 取 128-255 —— "A 按 M 切半" 就是这一步
    # composition 自动完成的，内核代码不出现任何 128 的手写偏移。
    # mma_g_c : ((128,256), 1, 1, 64, 64, 1)，C 半块同理。
    # SFB 走 mma_sfb（1CTA），partition_B 不切 N，保留完整 256。
    mma_thr = mma.get_slice(mma_tile_coord_v)
    mma_sfb_thr = mma_sfb.get_slice(mma_tile_coord_v)
    mma_g_a = mma_thr.partition_A(l_a)
    mma_g_b = mma_thr.partition_B(l_b)
    mma_g_sfa = mma_thr.partition_A(l_sfa)
    mma_g_sfb = mma_sfb_thr.partition_B(l_sfb)
    mma_g_c = mma_thr.partition_C(l_c)

    # Pair each staged SMEM tensor with its TMA coordinate tensor.
    #
    # tma_partition 做两件 layout 运算：
    # 1. group_modes(s_a, 0, 3)：把 ((128,32),1,4, stage) 的前三个模式折叠成
    #    一个 "一个 stage 的全部元素" 模式 -> (16384, 6)。GMEM 侧同样折叠成
    #    (tile 元素, tile_m, tile_k, l)。折叠是纯 shape 变形，stride 不变。
    # 2. 按 cta_coord 和 cluster 内 CTA 数把 box 再除一次（multicast 时每个
    #    CTA 只负责 box 的 1/N）。这里 K 方向 cluster 尺寸为 1，除法退化，
    #    每个 CTA 搬完整 box。
    # 结果 tma_s_a : (bulk 元素, stage)、tma_g_a : (bulk 元素, tile 坐标...)，
    # 二者第 0 模式一一对应——这正是 cute.copy 只需 (None, k_tile) ->
    # (None, stage) 两个下标就能发 TMA 的原因：坐标对齐已在 layout 里做完。
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
    # SF layout 里大量 stride=0 的模式（广播：32 个 K 元素共享一个 scale）。
    # filter_zeros 删除所有 stride=0 的模式，把 (((32,4),1),(32,1)) 压缩成
    # 真正占字节的紧凑形状——TMA 需要每个坐标对应唯一字节，不能有广播模式。
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
    #
    # make_fragment_A(s_a) 不产生寄存器：它把 s_a 的
    # (swizzle, 基址, stride) 编码成 64-bit SMEM matrix descriptor 的迭代器，
    # 形状仍是 ((M,K) atom, M重复=1, K重复=4, stage=6)。后面
    # r_a[(None,None,kblock,stage)] 的 "下标" 实际是在 descriptor 上加
    # 字节偏移（kblock 步进 32 B、stage 步进 16 KiB），swizzle 位保持 128B。
    # partition_shape_C((256,256)) -> ((128,256),1,1)：每 CTA 的 TMEM 半块。
    # 两个物理 accumulator view 的起点相差 208 列，而不是 256 列。它们
    # 重叠 48 列；epilogue 只有在完整 TMEM->RMEM copy fence 后才提前释放，
    # 因此下一 tile 覆盖重叠区时，旧 tile 已不再读取 TMEM。
    r_a = mma.make_fragment_A(s_a)
    r_b = mma.make_fragment_B(s_b)
    acc_shape = mma.partition_shape_C(MMA_TILE[:2])
    fake_acc = mma.make_fragment_C(cute.append(acc_shape, ACC_PHYSICAL_STAGES))
    fake_acc = cute.make_tensor(
        fake_acc.iterator,
        cute.make_layout(
            fake_acc.shape,
            stride=(
                fake_acc.stride[0],
                fake_acc.stride[1],
                fake_acc.stride[2],
                ACC_STAGE_STRIDE * fake_acc.stride[0][1],
            ),
        ),
    )

    # TMA multicast masks and the inverse mask used by tcgen05.commit to mark
    # every producer's local SMEM stage empty again.
    #
    # mask 也是 layout 运算：create_tma_multicast_mask 固定 cta_coord 的其余
    # 模式、沿 mcast_mode 扫一遍 cluster layout，把得到的每个 CTA rank 置进
    # 一个 16-bit 位图。
    #   cluster_layout_vmnk = ((2),1,1,1)，mode=1(N)/mode=2(K) 尺寸都是 1，
    #   所以 a/b/sfa mask 都只含自己 -> CTA0=0b01, CTA1=0b10（不做 multicast）。
    #   sfb 用 sfb_vmnk = ((1),2,1,1) 沿 mode=1（尺寸 2）扫 -> 0b11，
    #   一次 TMA 同时写两个 CTA 的 SMEM —— SFB "不切" 的另一半实现。
    # ab_empty_mask = 自己 | peer = 0b11：leader 的一条 tcgen05.commit 据此
    # 同时投递到两个 CTA 的 empty barrier。
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

    # The logical grid contains every output tile.  Resident clusters cancel
    # queued clusters through CLC and keep consuming their work, dynamically
    # balancing the tail without rebuilding per-tile resources.
    tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
        tile_sched_params,
        cute.arch.block_idx(),
        cute.arch.grid_dim(),
        clc_response_ptr,
    )
    work_tile = tile_sched.initial_work_tile_info()

    # TMEM allocation and all mbarrier initialization are a one-time persistent
    # prologue.  They remain live across every output tile owned by this cluster.
    tmem.allocate(TMEM_COLUMNS)
    if is_iket_cluster:
        cute.experimental.iket.range_pop()  # v13_prologue

    # CTA 0's scheduler warp stays one work item ahead of all compute roles.
    # It also consumes each response before reusing the single pipeline slot.
    if warp == SCHED_WARP and is_first_cta_in_cluster:
        clc_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.ProducerConsumer, 1
        )
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(clc_producer_state)
            clc_barrier = clc_pipeline.producer_get_barrier(
                clc_producer_state
            )
            tile_sched.advance_to_next_work(clc_barrier)
            clc_producer_state.advance()

            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            with cute.arch.elect_one():
                clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        clc_pipeline.producer_tail(clc_producer_state)

    # TMA producer.  AB stage/phase use a global K-tile count so the ring flows
    # directly across output-tile boundaries; there is only one final tail.
    if warp == TMA_WARP:
        work_idx = cutlass.Int32(0)
        if is_iket_cluster:
            cute.experimental.iket.range_push("tma_persistent_main")
        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            tma_g_a_tile = tma_g_a[
                (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
            ]
            tma_g_b_tile = tma_g_b[
                (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
            ]
            tma_g_sfa_tile = tma_g_sfa[
                (None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])
            ]
            tma_g_sfb_tile = tma_g_sfb[
                (None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])
            ]
            if is_iket_cluster:
                cute.experimental.iket.range_push("tma_work_tile")

            for k_tile in cutlass.range(K_TILES, unroll=1):
                linear_k_tile = cutlass.Uint32(work_idx * K_TILES + k_tile)
                stage = cutlass.Int32(linear_k_tile % AB_STAGES)
                generation = cutlass.Int32((linear_k_tile // AB_STAGES) % 2)
                empty_phase = generation ^ 1
                if is_iket_cluster:
                    cute.experimental.iket.range_push("tma_wait_empty")
                cute.arch.mbarrier_wait(ab_empty_mbar + stage, empty_phase)
                if is_iket_cluster:
                    cute.experimental.iket.range_pop()
                    cute.experimental.iket.range_push("tma_issue")

                if is_leader_cta:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full_mbar + stage, tma_bytes
                        )
                cute.copy(
                    tma_a,
                    tma_g_a_tile[(None, k_tile)],
                    tma_s_a[(None, stage)],
                    tma_bar_ptr=ab_full_mbar + stage,
                    mcast_mask=a_full_mask,
                )
                cute.copy(
                    tma_b,
                    tma_g_b_tile[(None, k_tile)],
                    tma_s_b[(None, stage)],
                    tma_bar_ptr=ab_full_mbar + stage,
                    mcast_mask=b_full_mask,
                )
                cute.copy(
                    tma_sfa,
                    tma_g_sfa_tile[(None, k_tile)],
                    tma_s_sfa[(None, stage)],
                    tma_bar_ptr=ab_full_mbar + stage,
                    mcast_mask=sfa_full_mask,
                )
                cute.copy(
                    tma_sfb,
                    tma_g_sfb_tile[(None, k_tile)],
                    tma_s_sfb[(None, stage)],
                    tma_bar_ptr=ab_full_mbar + stage,
                    mcast_mask=sfb_full_mask,
                )

                if is_iket_cluster:
                    cute.experimental.iket.range_pop()  # tma_issue

            if is_iket_cluster:
                cute.experimental.iket.range_pop()  # tma_work_tile
            work_idx += 1
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            with cute.arch.elect_one():
                clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()

        if is_iket_cluster:
            cute.experimental.iket.range_push("tma_tail")
        total_k_tiles = cutlass.Uint32(work_idx * K_TILES)
        for tail_offset in range(AB_STAGES):
            linear_stage = total_k_tiles + cutlass.Uint32(tail_offset)
            stage = cutlass.Int32(linear_stage % AB_STAGES)
            empty_phase = cutlass.Int32(
                1 ^ ((linear_stage // AB_STAGES) % 2)
            )
            cute.arch.mbarrier_wait(ab_empty_mbar + stage, empty_phase)
        if is_iket_cluster:
            cute.experimental.iket.range_pop()  # tma_tail
            cute.experimental.iket.range_pop()  # tma_persistent_main

    # MMA consumer.  The two physical accumulator views and completion
    # barriers alternate by work index; one shared empty barrier serializes
    # exactly the 48-column cross-view overlap.
    if warp == MMA_WARP:
        work_idx = cutlass.Int32(0)
        if is_iket_cluster:
            cute.experimental.iket.range_push("mma_persistent_main")
            cute.experimental.iket.range_push("mma_tmem_wait")
        tmem.wait_for_alloc()
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
        acc_ptr = tmem.retrieve_ptr(F32)
        acc_base = cute.make_tensor(acc_ptr, fake_acc.layout)

        # These descriptor objects are constructed outside the dynamic
        # leader-CTA branch.  CuTe DSL gives staged if/while regions lexical
        # SSA scope, so defining them only in ``if is_leader_cta`` would make
        # them unavailable to the persistent loop below.  Only the leader CTA
        # executes the S2T copies and MMAs; constructing the descriptors is
        # compile-time address/layout algebra and emits no transport itself.
        sfa_ptr = cute.recast_ptr(acc_ptr + ACC_TMEM_COLUMNS, dtype=SF8)
        t_sfa_layout = blockscaled_utils.make_tmem_layout_sfa(
            mma,
            MMA_TILE,
            SF_VECTOR_SIZE,
            cute.slice_(s_layout_sfa, (None, None, None, 0)),
        )
        t_sfb_layout = blockscaled_utils.make_tmem_layout_sfb(
            mma,
            MMA_TILE,
            SF_VECTOR_SIZE,
            cute.slice_(s_layout_sfb, (None, None, None, 0)),
        )
        t_sfa = cute.make_tensor(sfa_ptr, t_sfa_layout)
        sfb_ptr = cute.recast_ptr(
            acc_ptr + ACC_TMEM_COLUMNS + SFA_TMEM_COLUMNS,
            dtype=SF8,
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

        while work_tile.is_valid_tile:
            acc_stage = work_idx % ACC_PHYSICAL_STAGES
            acc = acc_base[(None, None, None, acc_stage)]
            if is_iket_cluster:
                cute.experimental.iket.range_push("mma_work_tile")

            if is_leader_cta:
                empty_phase = 1 ^ (work_idx % 2)
                if is_iket_cluster:
                    cute.experimental.iket.range_push("mma_wait_acc_empty")
                cute.arch.mbarrier_wait(acc_empty_mbar, empty_phase)
                if is_iket_cluster:
                    cute.experimental.iket.range_pop()

                mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(K_TILES, unroll=1):
                    linear_k_tile = cutlass.Uint32(
                        work_idx * K_TILES + k_tile
                    )
                    stage = cutlass.Int32(linear_k_tile % AB_STAGES)
                    full_phase = cutlass.Int32(
                        (linear_k_tile // AB_STAGES) % 2
                    )
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
                        cute.experimental.iket.range_push("mma_release_ab")
                    with cute.arch.elect_one():
                        tcgen05.commit(
                            ab_empty_mbar + stage,
                            ab_empty_mask,
                            tcgen05.CtaGroup.TWO,
                        )
                    if is_iket_cluster:
                        cute.experimental.iket.range_pop()  # mma_release_ab

                if is_iket_cluster:
                    cute.experimental.iket.range_push("mma_commit_acc")
                with cute.arch.elect_one():
                    tcgen05.commit(
                        acc_full_mbar + acc_stage,
                        acc_full_mask,
                        tcgen05.CtaGroup.TWO,
                    )
                if is_iket_cluster:
                    cute.experimental.iket.range_pop()

            if is_iket_cluster:
                cute.experimental.iket.range_pop()  # mma_work_tile
            work_idx += 1
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            with cute.arch.elect_one():
                clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()

        if is_leader_cta:
            if is_iket_cluster:
                cute.experimental.iket.range_push("mma_acc_tail")
            final_empty_phase = 1 ^ (work_idx % 2)
            cute.arch.mbarrier_wait(acc_empty_mbar, final_empty_phase)
            if is_iket_cluster:
                cute.experimental.iket.range_pop()
        if is_iket_cluster:
            cute.experimental.iket.range_pop()  # mma_persistent_main

    # Four epilogue warps drain 128x64 subtiles directly from TMEM through
    # registers to GMEM.  Each thread holds only 64 FP32 accumulator values at
    # once, instead of the whole 256-column fragment used by v12.  After the
    # overlapping columns have left TMEM, the accumulator is released while
    # the remaining conversion/stores run.
    if warp < MMA_WARP:
        work_idx = cutlass.Int32(0)
        if is_iket_cluster:
            cute.experimental.iket.range_push("epi_persistent_main")
            cute.experimental.iket.range_push("epi_tmem_wait")
        tmem.wait_for_alloc()
        if is_iket_cluster:
            cute.experimental.iket.range_pop()
        acc_ptr = tmem.retrieve_ptr(F32)
        acc_base = cute.make_tensor(acc_ptr, fake_acc.layout)

        acc_epi = cute.flat_divide(
            acc_base[((None, None), 0, 0, None)], epi_tile
        )
        t2r_atom = sm100_utils.get_tmem_load_op(
            CTA_TILE,
            utils.LayoutEnum.ROW_MAJOR,
            BF16,
            F32,
            epi_tile,
            True,
        )
        t2r = tcgen05.make_tmem_copy(
            t2r_atom, acc_epi[(None, None, 0, 0, 0)]
        )
        t2r_thr = t2r.get_slice(tid)
        t_acc_base = t2r_thr.partition_S(acc_epi)
        g_c_epi = cute.flat_divide(
            mma_g_c[((None, None), 0, 0, None, None, None)], epi_tile
        )
        t_g_c = t2r_thr.partition_D(g_c_epi)
        r_acc = cute.make_rmem_tensor(
            t_g_c[(None, None, None, 0, 0, 0, 0, 0)].shape, F32
        )
        r_c = cute.make_rmem_tensor(r_acc.shape, BF16)
        common_layout = cute.max_common_layout(
            r_c.layout,
            t_g_c[(None, None, None, 0, 0, 0, 0, 0)].layout,
        )
        gmem_copy_bits = min(
            t_g_c.iterator.alignment * 8,
            cute.size(common_layout) * BF16.width,
            256,
        )
        gmem_store = cute.make_copy_atom(
            cute.nvgpu.CopyR2GOp(),
            BF16,
            num_bits_per_copy=gmem_copy_bits,
            l1c_evict_priority=CacheEvictionPriority.NO_ALLOCATE,
        )

        early_release_subtile = (
            cute.ceil_div(SF_TMEM_COLUMNS, cute.size(epi_tile[1])) - 1
        )

        while work_tile.is_valid_tile:
            cur_tile_coord = work_tile.tile_idx
            mma_tile_coord_mnl = (
                cur_tile_coord[0] // cute.size(mma.thr_id.shape),
                cur_tile_coord[1],
                cur_tile_coord[2],
            )
            acc_stage = work_idx % ACC_PHYSICAL_STAGES
            full_phase = (work_idx // ACC_FULL_STAGES) % 2
            reverse_subtile = acc_stage == 0
            t_acc = t_acc_base[
                (None, None, None, None, None, acc_stage)
            ]
            out = t_g_c[
                (None, None, None, None, None, *mma_tile_coord_mnl)
            ]
            if is_iket_cluster:
                cute.experimental.iket.range_push("epi_work_tile")
                cute.experimental.iket.range_push("epi_wait_acc")
            cute.arch.mbarrier_wait(
                acc_full_mbar + acc_stage, full_phase
            )
            if is_iket_cluster:
                cute.experimental.iket.range_pop()

            t_acc = cute.group_modes(t_acc, 3, cute.rank(t_acc))
            out = cute.group_modes(out, 3, cute.rank(out))
            subtile_count = cute.size(t_acc.shape, mode=[3])
            for subtile_idx in range(subtile_count):
                real_subtile_idx = subtile_idx
                if reverse_subtile:
                    real_subtile_idx = subtile_count - 1 - subtile_idx

                if is_iket_cluster:
                    cute.experimental.iket.range_push("epi_t2r")
                cute.copy(
                    t2r,
                    t_acc[(None, None, None, real_subtile_idx)],
                    r_acc,
                )
                if subtile_idx == early_release_subtile:
                    cute.arch.fence_view_async_tmem_load()
                    if is_iket_cluster:
                        cute.experimental.iket.range_pop()
                        cute.experimental.iket.range_push("epi_early_release_acc")
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(
                            acc_empty_mbar, acc_empty_dst_rank
                        )
                    if is_iket_cluster:
                        cute.experimental.iket.range_pop()
                elif is_iket_cluster:
                    cute.experimental.iket.range_pop()

                r_c.store(r_acc.load().to(BF16))
                cute.copy(
                    gmem_store,
                    r_c,
                    out[(None, None, None, real_subtile_idx)],
                )

            if is_iket_cluster:
                cute.experimental.iket.range_pop()  # epi_work_tile

            work_idx += 1
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            with cute.arch.elect_one():
                clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()

        if is_iket_cluster:
            cute.experimental.iket.range_pop()  # epi_persistent_main

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
        cute.experimental.iket.range_pop()  # dense_gemm_v13


@cute.jit
def dense_gemm_v13(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    max_active_clusters: cutlass.Constexpr,
    stream,
):
    """Build and launch the fixed persistent 2SM MXFP8 GEMM."""

    a = cute.make_tensor(a_ptr, cute.make_layout((M, K, 1), stride=(K, 1, M * K)))
    b = cute.make_tensor(b_ptr, cute.make_layout((N, K, 1), stride=(K, 1, N * K)))
    c = cute.make_tensor(c_ptr, cute.make_layout((M, N, 1), stride=(N, 1, M * N)))

    # sf_atom = cute.make_layout(
    #      M 维度     K 维度
    #     ((32, 4), (SF_VECTOR_SIZE, 4)),
    #     stride=((16, 4), (0, 1)),
    # )
    #   SF 不做 swizzle，但必须是 Blackwell 规定的 32x4x4 canonical 排布
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
    problem_shape_ntile_mnl = (
        M // CTA_TILE[0],
        N // CTA_TILE[1],
        1,
    )
    tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
        problem_shape_ntile_mnl,
        CLUSTER_SHAPE,
        CLUSTER_SWIZZLE_SIZE,
        RASTER_ALONG_M,
    )
    grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(
        tile_sched_params
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
    #   ((128,32), 1, 4, 6)
    # - s_layout_a = tcgen05.tile_to_mma_shape(
    #       a_smem_layout_atom,
    #       a_smem_shape,
    #       order=order,
    #   )
    # S<3,4,3> o 0 o
    # ((128,32),1,4,6):((128,1),0,32,16384)
    s_layout_a = sm100_utils.make_smem_layout_a(mma, MMA_TILE, FP8, AB_STAGES)
    s_layout_b = sm100_utils.make_smem_layout_b(mma, MMA_TILE, FP8, AB_STAGES)
    # s_layout_sfa =
    # ((((32,4),1),(32,1)),1,4,6):
    # ((((16,4),0),(0,0)),0,1,512)
    #
    # s_layout_sfb =
    # ((((32,4),2),(32,1)),1,4,6):
    # ((((16,4),512),(0,0)),0,1,1024)
    #
    # 逐模式拆解。SF atom：((32,4),(32,1)) : ((16,4),(0,1))
    #   外层 (32,4) : (16,4)   M 方向的 (atom 内 32 行, 4 个行组)
    #   内层 (32,1) : (0,1)    K 方向的 (32 个 scale, 1) —— stride 0 是
    #                          广播：atom 的 K 实际只占 1 列、32 元素共享。
    #                   （stride 0 的模式在 TMA/s2t 前必须 filter_zeros。）
    # atom 本身：32 行 x 1 列 = 32 B，但逻辑上是 32 行 x 32 K 元素。
    #
    # 平铺到完整 tile（tile_to_mma_shape 的"填盒"）：读 stride 即可得
    # 单 stage 的地址函数（滤掉 stride=0 的广播模式后）：
    #   SFA: byte(m, kb) = (m%32)*16 + (m//32)*4 + kb*1
    #        m_lo 一步 16 B（一"行"），行内 16 B = 4 行组 x 4 kblock，
    #        双射到 0..511。
    #   SFB: 同式再加 (n//128)*512 —— N=256 拆成两个 128 半块，
    #        第二半块整体偏移 512 B（shape 里那个 2、stride 512）。
    # stage 步长：SFA 512 B / SFB 1024 B。
    #
    # 与 TMEM 侧的对应（kernel 里 t_sfa/t_sfb 注释）：
    #   SMEM SFA  512 B /stage  -> 紧凑 128 字节 <-> TMEM 16 列
    #   SMEM SFB 1024 B/stage  -> 紧凑 256 字节 <-> TMEM 32 列
    #   两者都 = (行数/32) * 4 列：SFA 128/32*4=16 列，SFB 256/32*4=32 列。
    #   TMEM 的物理排布仍由 make_tmem_layout_sfa/sfb 生成，本机不可 print
    #   验证；SMEM→TMEM 之间的真实对比建议在设备上 cute.printf 后核对。
    s_layout_sfa = blockscaled_utils.make_smem_layout_sfa(
        mma, MMA_TILE, SF_VECTOR_SIZE, AB_STAGES
    )
    s_layout_sfb = blockscaled_utils.make_smem_layout_sfb(
        mma, MMA_TILE, SF_VECTOR_SIZE, AB_STAGES
    )
    # Use 128x64 for the bulk direct-store path.  Passing source-C metadata is
    # only a sizing hint; this kernel does not load a source C tensor.
    epi_tile = sm100_utils.compute_epilogue_tile_shape(
        CTA_TILE,
        True,
        utils.LayoutEnum.ROW_MAJOR,
        BF16,
        layout_c=utils.LayoutEnum.ROW_MAJOR,
        elem_ty_c=BF16,
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
    #
    # 这里的 layout 运算是 "用 SMEM stage layout 反推 TMA box"：
    # 传入的 slice_(s_layout_a, (...,0)) = S<3,4,3> o ((128,32),1,4):((128,1),0,32)
    # helper 从中读出 (a) box 形状 128x128，(b) swizzle 枚举 = 128B（来自
    # S<3,4,3>），(c) 内维必须连续 128 B 的合法性检查，然后写进 tensor map。
    # 返回的 g_a 不是数据张量而是 "坐标张量"：shape 同 mma_g_a 的分块结果，
    # 元素是 TMA 坐标，供 tma_partition/copy 消费。
    # SF 侧 internal_type=Int16：SF8 单字节不满足 TMA 元素粒度要求，tensor
    # map 用 16-bit 视图描述（filter_zeros 后的紧凑 layout 保证可整除）。
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

    _dense_gemm_v13_kernel(
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
        tile_sched_params,
        s_layout_a,
        s_layout_b,
        s_layout_sfa,
        s_layout_sfb,
        epi_tile,
        tma_bytes,
    ).launch(
        grid=grid,
        block=(THREADS, 1, 1),
        cluster=CLUSTER_SHAPE,
        stream=stream,
        min_blocks_per_mp=1,
    )


__all__ = [
    "AB_STAGES",
    "ACC_STAGES",
    "ACC_FULL_STAGES",
    "BF16",
    "CLUSTER_SHAPE",
    "CLUSTER_SWIZZLE_SIZE",
    "CLUSTER_TILES_M",
    "CLUSTER_TILES_N",
    "CTA_TILE",
    "EPILOGUE_WARPS",
    "F32",
    "FP8",
    "IKET_PERSISTENT_CLUSTER",
    "MMA_INSTRUCTION",
    "MMA_TILE",
    "MMA_WARP",
    "OUTPUT_CLUSTER_TILES",
    "RASTER_ALONG_M",
    "SCHED_WARP",
    "SFB_MMA_TILE",
    "SF8",
    "SFA_TMEM_COLUMNS",
    "SFB_TMEM_COLUMNS",
    "SF_TMEM_COLUMNS",
    "ACC_PHYSICAL_STAGES",
    "ACC_STAGE_STRIDE",
    "ACC_TMEM_COLUMNS",
    "SF_VECTOR_SIZE",
    "THREADS",
    "TMA_WARP",
    "dense_gemm_v13",
]
