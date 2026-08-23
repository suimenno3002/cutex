"""Blank CuTeDSL scaffold for the fixed B300 MXFP8 Fprop contract."""

import cutlass
import cutlass.cute as cute

from .dense_gemm_contract import MMA_SHAPE, SF_VECTOR_SIZE, TILE_K, TILE_M, TILE_N


FP8 = cutlass.Float8E4M3FN
SF8 = cutlass.Float8E8M0FNU
F32 = cutlass.Float32
BF16 = cutlass.BFloat16
TILE = (TILE_M, TILE_N, TILE_K)
THREADS, AB_STAGES, ACC_STAGES = 128, 4, 1


@cute.kernel
def _dense_gemm_v0_kernel(
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
    # TODO: 建立 CTA 坐标、IKET ranges、SMEM/TMEM 与两组异步 pipeline。
    # TODO: TMA 搬运 FP8 A/B 和 E8M0 SFA/SFB；scale 还需从 SMEM 复制到 TMEM。
    # TODO: 为每个 K block 设置 Field.SFA/SFB，发射 MXFP8 MMA 到 FP32 TMEM。
    # TODO: TMEM -> registers，转换成 BF16 后写回 Y[16384,16384]。
    # TODO: 完成 barrier、pipeline state 与 TMEM 生命周期管理。
    raise NotImplementedError("implement _dense_gemm_v0_kernel")


@cute.jit
def dense_gemm_v0(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    sfa_ptr: cute.Pointer,
    sfb_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    stream,
):
    """Implement the fixed ``BF16 Y = MXFP8 X @ W.T`` launch here.

    ``a_ptr``/``b_ptr`` are row-major E4M3FN ``[16384,16384]``. ``sfa_ptr`` and
    ``sfb_ptr`` are E8M0FNU scales in tcgen05 Swizzle32x4x4 layout, one per 32
    contiguous K values. The MMA must accumulate in FP32 and store BF16.
    """

    # TODO: 用固定三维逻辑布局 (M,K,1)/(N,K,1)/(M,N,1) 包装五个 pointer。
    # TODO: 构造 MmaMXF8F6F4Op(*MMA_SHAPE)，A/B 都设为 K-major。
    # TODO: 构造 A/B 与 SFA/SFB 的 staged SMEM layouts 和四个 TMA atoms。
    # TODO: 以 TILE=(128,256,128)、block=(128,1,1) 启动上面的 kernel。
    raise NotImplementedError("implement dense_gemm_v0")


__all__ = [
    "AB_STAGES",
    "ACC_STAGES",
    "BF16",
    "F32",
    "FP8",
    "MMA_SHAPE",
    "SF8",
    "SF_VECTOR_SIZE",
    "THREADS",
    "TILE",
    "dense_gemm_v0",
]
