import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "cutex" / "kernels" / "dense_gemm_v2.py"
)


def _source():
    return SOURCE_PATH.read_text(encoding="utf-8")


def _tree():
    return ast.parse(_source())


def _function(name):
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_dense_gemm_v2_uses_four_tma_loads_without_pipeline_or_mma():
    source = _source()
    names = {node.id for node in ast.walk(_tree()) if isinstance(node, ast.Name)}
    assert source.count("cpasync.tma_partition(") == 4
    assert source.count("tma_bar_ptr=tma_mbar") == 4
    assert "PipelineTma" not in source
    # Layout documentation may mention the tcgen05 hardware contract; v2 must
    # not import or reference tcgen05 in executable Python code.
    assert "tcgen05" not in names
    assert "cute.gemm(" not in source


def test_dense_gemm_v2_uses_manual_single_stage_mbarrier_protocol():
    source = _source()
    assert "THREADS, TMA_STAGES = 256, 1" in source
    init = source.index("cute.arch.mbarrier_init(tma_mbar, 1)")
    arm = source.index("cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, tma_bytes)")
    issue = source.index("tma_bar_ptr=tma_mbar")
    wait = source.index("cute.arch.mbarrier_wait(tma_mbar, phase)")
    assert init < arm < issue < wait
    assert "phase = phase ^ cutlass.Int32(1)" in source
    assert source.count("cute.arch.sync_threads()") >= 2


def test_dense_gemm_v2_loads_native_scale_blocks_through_tma():
    source = _source()
    assert "SCALE_TILE = (128, 128)" in source
    assert source.count("blockscaled_utils.tile_atom_to_shape_SF(") == 2
    assert "cute.tile_to_shape(sf_atom, a.shape, (2, 1, 3))" in source
    assert "cute.tile_to_shape(sf_atom, b.shape, (2, 1, 3))" in source
    assert 'print("[dense_gemm_v2] sfa.layout =", sfa.layout)' in source
    assert 'print("[dense_gemm_v2] sfb.layout =", sfb.layout)' in source
    assert "#   shape  = (((32, 4), 128), ((32, 4), 128), 1)" in source
    assert (
        "#   stride = (((16, 4), 65536), ((0, 1), 512), 8388608)" in source
    )
    assert source.count("blockscaled_utils.make_smem_layout_sf(") == 2
    assert 'print("[dense_gemm_v2] s_layout_sfa =", s_layout_sfa)' in source
    assert 'print("[dense_gemm_v2] s_layout_sfb =", s_layout_sfb)' in source
    assert "# sf_atom = cute.make_layout(" in source
    assert "# sf_tile_layout = cute.tile_to_shape(" in source
    assert "#     stride=cute.cosize(cute.filter_zeros(sf_tile_layout))" in source
    assert "#   shape  = ((32, 4), (32, 4), 1)" in source
    assert "#   stride = ((16, 4), (0, 1), 512)" in source
    assert source.count("internal_type=cutlass.Int16") == 2
    assert "tma_g_sfa[(None, k_tile)]" in source
    assert "tma_g_sfb[(None, k_tile)]" in source


def test_dense_gemm_v2_does_not_directly_load_gmem_operands():
    kernel = _function("_dense_gemm_v2_kernel")
    operand_names = {"g_a", "g_b", "g_sfa", "g_sfb"}
    direct_loads = [
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id in operand_names
    ]
    assert not direct_loads


def test_dense_gemm_v2_uses_fp32_scalar_accumulation_and_bf16_store():
    source = _source()
    assert "acc = F32(0.0)" in source
    assert "acc = acc + (a * sa) * (b * sb)" in source
    assert "acc.to(BF16)" in source


def test_dense_gemm_v2_preserves_the_five_pointer_launch_interface():
    function = _function("dense_gemm_v2")
    assert [argument.arg for argument in function.args.args] == [
        "a_ptr",
        "b_ptr",
        "sfa_ptr",
        "sfb_ptr",
        "c_ptr",
        "stream",
    ]
