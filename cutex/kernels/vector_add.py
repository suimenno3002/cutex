"""Autotuned A+B CuTeDSL kernel based on NVIDIA's canonical TV-layout pattern."""

import cutlass
import cutlass.cute as cute

from cutex.autotune import Config, autotune


VECTOR_ADD_CONFIGS = [
    Config(kwargs={"copy_bits": 32}, name="scalar-32b"),
    Config(kwargs={"copy_bits": 64}, name="vector-64b"),
    Config(kwargs={"copy_bits": 128}, name="vector-128b"),
]


@cute.kernel
def vector_add_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
    coordinates: cute.Tensor,
    shape: cute.Shape,
    thread_layout: cute.Layout,
    value_layout: cute.Layout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    cute.experimental.iket.range_push("vector_add")
    cute.experimental.iket.range_push("setup")

    block_coord = ((None, None), block_idx)
    block_a = g_a[block_coord]
    block_b = g_b[block_coord]
    block_c = g_c[block_coord]
    block_coordinates = coordinates[block_coord]

    load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), g_a.element_type)
    store_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), g_c.element_type)
    tiled_a = cute.make_tiled_copy_tv(load_atom, thread_layout, value_layout)
    tiled_b = cute.make_tiled_copy_tv(load_atom, thread_layout, value_layout)
    tiled_c = cute.make_tiled_copy_tv(store_atom, thread_layout, value_layout)

    thread_a = tiled_a.get_slice(thread_idx).partition_S(block_a)
    thread_b = tiled_b.get_slice(thread_idx).partition_S(block_b)
    thread_c = tiled_c.get_slice(thread_idx).partition_S(block_c)
    fragment_a = cute.make_rmem_tensor_like(thread_a)
    fragment_b = cute.make_rmem_tensor_like(thread_b)
    fragment_c = cute.make_rmem_tensor_like(thread_c)

    thread_coordinates = tiled_c.get_slice(thread_idx).partition_S(block_coordinates)
    predicate = cute.make_rmem_tensor(thread_coordinates.shape, cutlass.Boolean)
    for index in range(cute.size(predicate)):
        predicate[index] = cute.elem_less(thread_coordinates[index], shape)

    cute.experimental.iket.range_pop()  # setup
    cute.experimental.iket.range_push("load")
    cute.copy(load_atom, thread_a, fragment_a, pred=predicate)
    cute.copy(load_atom, thread_b, fragment_b, pred=predicate)
    cute.experimental.iket.range_pop()
    cute.experimental.iket.range_push("add")
    fragment_c.store(fragment_a.load() + fragment_b.load())
    cute.experimental.iket.range_pop()
    cute.experimental.iket.range_push("store")
    cute.copy(store_atom, fragment_c, thread_c, pred=predicate)
    cute.experimental.iket.range_pop()
    cute.experimental.iket.range_pop()  # vector_add


@autotune(
    configs=VECTOR_ADD_CONFIGS,
    key=["m", "n"],
    warmup=5,
    rep=30,
    cache_results=True,
)
@cute.jit
def vector_add(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    m: cutlass.Int32,
    n: cutlass.Int32,
    copy_bits: cutlass.Constexpr = 128,
):
    # m/n are runtime autotune keys; the tensor layout itself drives the launch.
    dtype = a.element_type
    vector_size = copy_bits // dtype.width

    thread_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    value_layout = cute.make_ordered_layout((4, vector_size), order=(1, 0))
    tile, tv_layout = cute.make_layout_tv(thread_layout, value_layout)

    g_a = cute.zipped_divide(a, tile)
    g_b = cute.zipped_divide(b, tile)
    g_c = cute.zipped_divide(c, tile)
    identity = cute.make_identity_tensor(c.shape)
    coordinates = cute.zipped_divide(identity, tiler=tile)

    vector_add_kernel(
        g_a,
        g_b,
        g_c,
        coordinates,
        c.shape,
        thread_layout,
        value_layout,
    ).launch(
        grid=[cute.size(g_c, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )
