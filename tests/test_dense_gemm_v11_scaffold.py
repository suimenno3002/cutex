import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "cutex" / "kernels" / "dense_gemm_v11.py"
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


def _literal_assignment(name):
    assignment = next(
        node
        for node in _tree().body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def test_v11_fixes_the_two_cta_tile_cluster_and_warp_roles():
    source = _source()
    assert _literal_assignment("MMA_INSTRUCTION") == (256, 256, 32)
    assert _literal_assignment("MMA_TILE") == (256, 256, 128)
    assert _literal_assignment("CTA_TILE") == (128, 256, 128)
    assert _literal_assignment("SFB_MMA_TILE") == (128, 256, 128)
    assert _literal_assignment("CLUSTER_SHAPE") == (2, 1, 1)
    assert _literal_assignment("CLUSTER_SWIZZLE_SIZE") == 8
    assert _literal_assignment("EPILOGUE_WARPS") == (0, 1, 2, 3)
    assert "AB_STAGES, ACC_STAGES = 6, 1" in source
    assert "MMA_WARP = 4" in source
    assert "TMA_WARP = 5" in source
    assert "THREADS = 6 * 32" in source


def test_v11_host_side_uses_mxfp8_two_cta_and_a_companion_sfb_mma():
    source = _source()
    assert "FP8 = cutlass.Float8E4M3FN" in source
    assert "SF8 = cutlass.Float8E8M0FNU" in source
    assert "F32 = cutlass.Float32" in source
    assert "BF16 = cutlass.BFloat16" in source
    assert source.count("sm100_utils.make_blockscaled_trivial_tiled_mma(") == 2
    assert "tcgen05.CtaGroup.TWO" in source
    assert "tcgen05.CtaGroup.ONE" in source
    assert source.count("OperandMajorMode.K") >= 4
    assert "sm100_utils.cluster_shape_to_tma_atom_SFB(" in source
    assert "mma_sfb" in source


def test_v11_builds_four_staged_tma_paths_without_pipeline_wrappers():
    source = _source()
    assert source.count("blockscaled_utils.tile_atom_to_shape_SF(") == 2
    assert "sm100_utils.make_smem_layout_a(" in source
    assert "sm100_utils.make_smem_layout_b(" in source
    assert "blockscaled_utils.make_smem_layout_sfa(" in source
    assert "blockscaled_utils.make_smem_layout_sfb(" in source
    assert source.count("cute.nvgpu.make_tiled_tma_atom_A(") == 2
    assert source.count("cute.nvgpu.make_tiled_tma_atom_B(") == 2
    assert source.count("internal_type=cutlass.Int16") == 2
    assert "import cutlass.pipeline as pipeline" not in source
    assert "pipeline.PipelineTmaUmma" not in source
    assert "pipeline.PipelineUmmaAsync" not in source
    assert source.count("cpasync.tma_partition(") == 4
    assert source.count("tma_bar_ptr=ab_full_mbar + stage") == 4


def test_v11_expands_full_empty_mbarrier_protocol_and_two_cta_signaling():
    source = _source()
    assert "ab_empty_mbar = ab_full_mbar + AB_STAGES" in source
    assert "acc_empty_mbar = acc_full_mbar + ACC_STAGES" in source
    assert "cute.arch.mbarrier_init(ab_full_mbar + stage, 1)" in source
    assert "cute.size(cluster_layout_vmnk, mode=[0]) * len(" in source
    assert "EPILOGUE_WARPS" in source
    assert "cute.arch.mbarrier_init_fence()" in source
    assert "cute.arch.cluster_arrive()" in source
    assert "cute.arch.cluster_wait()" in source
    assert "cute.arch.mbarrier_arrive_and_expect_tx(" in source
    assert source.count("tcgen05.commit(") == 2
    assert "ab_empty_mask" in source
    assert "acc_full_mask" in source
    assert "with cute.arch.elect_one():" in source
    assert "cute.arch.mbarrier_arrive(acc_empty_mbar, acc_empty_dst_rank)" in source


def test_v11_manually_tracks_ring_stage_phase_and_drains_tail():
    source = _source()
    assert "stage = k_tile % AB_STAGES" in source
    assert "generation = (k_tile // AB_STAGES) % 2" in source
    assert "empty_phase = generation ^ 1" in source
    assert "full_phase = (k_tile // AB_STAGES) % 2" in source
    assert "for tail_offset in range(AB_STAGES):" in source
    assert "linear_stage = K_TILES + tail_offset" in source
    assert "empty_phase = 1 ^ ((linear_stage // AB_STAGES) % 2)" in source


def test_v11_is_a_regular_grid_without_a_persistent_tile_scheduler():
    source = _source()
    tree = _tree()
    assert not any(isinstance(node, ast.While) for node in ast.walk(tree))
    assert "utils.StaticPersistentTileScheduler" not in source
    assert "utils.ClcDynamicPersistentTileScheduler" not in source
    assert "grid=(M // CTA_TILE[0], N // CTA_TILE[1], 1)" in source
    assert "cluster=CLUSTER_SHAPE" in source


def test_v11_has_warp_specialized_device_schedule_and_direct_store_epilogue():
    source = _source()
    assert "if warp == TMA_WARP:" in source
    assert "if warp == MMA_WARP:" in source
    assert "if warp < MMA_WARP:" in source
    assert "raise NotImplementedError(" not in source
    assert "TODO(v11." not in source
    assert "r_out.store(r_acc.load().to(BF16))" in source
    assert "cute.nvgpu.CopyUniversalOp()" in source
    assert "cute.arch.barrier()" in source


def test_v11_exposes_role_and_stall_ranges_to_iket():
    source = _source()
    assert _literal_assignment("IKET_CLUSTER") == (32, 32, 0)
    assert "is_iket_cluster = cluster_linear == iket_cluster_linear" in source
    expected_ranges = {
        "dense_gemm_v11",
        "v11_prologue",
        "tma_main",
        "tma_k_tile",
        "tma_wait_empty",
        "tma_issue",
        "tma_tail",
        "mma_main",
        "mma_tmem_wait",
        "mma_wait_acc_empty",
        "mma_k_tile",
        "mma_wait_ab_full",
        "mma_s2t",
        "mma_issue",
        "mma_release_ab",
        "mma_commit_acc",
        "mma_wait_acc_empty_tail",
        "epi_main",
        "epi_tmem_wait",
        "epi_setup",
        "epi_wait_acc",
        "epi_t2r",
        "epi_store",
        "epi_release_acc",
        "cta_tail_sync",
        "tmem_dealloc",
    }
    for range_name in expected_ranges:
        assert f'range_push("{range_name}"' in source


def test_v11_preserves_the_five_pointer_launch_interface():
    function = _function("dense_gemm_v11")
    assert [argument.arg for argument in function.args.args] == [
        "a_ptr",
        "b_ptr",
        "sfa_ptr",
        "sfb_ptr",
        "c_ptr",
        "stream",
    ]


def test_v11_uses_a_cute_layout_for_an_eight_by_eight_cluster_block_swizzle():
    """Mirror the inverse of the fixed CuTe layout and prove its cluster-tile
    mapping is bijective and compact for a roughly one-wave launch window."""
    source = _source()
    swizzle_size = 8
    cluster_grid_m = 64

    def decode(rank):
        n_inner = rank % swizzle_size
        quotient = rank // swizzle_size
        tile_m = quotient % cluster_grid_m
        tile_n = (quotient // cluster_grid_m) * swizzle_size + n_inner
        return tile_m, tile_n

    tiles = {decode(cid) for cid in range(64 * 64)}
    assert len(tiles) == 64 * 64
    assert {decode(cid) for cid in range(64)} == {
        (tile_m, tile_n) for tile_m in range(8) for tile_n in range(8)
    }

    windows = []
    for base in range(0, 64 * 64 - 74, 74):
        win = [decode(cid) for cid in range(base, base + 74)]
        ms = [t[0] for t in win]
        ns = [t[1] for t in win]
        windows.append((max(ms) - min(ms) + 1, max(ns) - min(ns) + 1))
    median_m = sorted(w[0] for w in windows)[len(windows) // 2]
    median_n = sorted(w[1] for w in windows)[len(windows) // 2]
    assert (median_m, median_n) == (10, 8)

    assert "ZORDER_BITS" not in source
    assert "_zorder_decode" not in source
    assert "cluster_tile_layout_mnl = cute.make_layout(" in source
    assert "(CLUSTER_SWIZZLE_SIZE, cluster_grid_n // CLUSTER_SWIZZLE_SIZE)" in source
    assert (
        "tile_m, tile_n, tile_l = cluster_tile_layout_mnl.get_flat_coord("
        in source
    )
