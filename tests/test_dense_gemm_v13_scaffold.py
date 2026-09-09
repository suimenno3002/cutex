import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "cutex" / "kernels" / "dense_gemm_v13.py"
)
REPO_ROOT = SOURCE_PATH.parents[2]


def _source():
    return SOURCE_PATH.read_text(encoding="utf-8")


def _tree():
    return ast.parse(_source())


def _literal_assignment(name):
    assignment = next(
        node
        for node in _tree().body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
    )
    return ast.literal_eval(assignment.value)


def _function(name):
    return next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def test_v13_fixed_tile_cluster_and_warp_roles():
    assert _literal_assignment("MMA_INSTRUCTION") == (256, 256, 32)
    assert _literal_assignment("MMA_TILE") == (256, 256, 128)
    assert _literal_assignment("CTA_TILE") == (128, 256, 128)
    assert _literal_assignment("SFB_MMA_TILE") == (128, 256, 128)
    assert _literal_assignment("CLUSTER_SHAPE") == (2, 1, 1)
    assert _literal_assignment("CLUSTER_SWIZZLE_SIZE") == 8
    assert _literal_assignment("RASTER_ALONG_M") is True
    assert _literal_assignment("ACC_FULL_STAGES") == 2
    assert _literal_assignment("EPILOGUE_WARPS") == (0, 1, 2, 3)
    source = _source()
    assert "AB_STAGES, ACC_STAGES = 6, 1" in source
    assert "MMA_KBLOCKS_PER_STAGE = MMA_TILE[2] // MMA_INSTRUCTION[2]" in source
    assert "MMA_WARP = 4" in source
    assert "TMA_WARP = 5" in source
    assert "SCHED_WARP = 6" in source
    assert "THREADS = 7 * 32" in source


def test_v13_uses_clc_dynamic_persistent_scheduling():
    source = _source()
    assert "pipeline.PipelineClcFetchAsync.create(" in source
    assert "utils.ClcDynamicPersistentTileSchedulerParams(" in source
    assert "utils.ClcDynamicPersistentTileScheduler.get_grid_shape(" in source
    assert "utils.ClcDynamicPersistentTileScheduler.create(" in source
    assert "tile_sched.advance_to_next_work(clc_barrier)" in source
    assert source.count("while work_tile.is_valid_tile:") == 4
    assert source.count("clc_pipeline.consumer_wait(clc_consumer_state)") == 4
    assert source.count("clc_pipeline.consumer_release(clc_consumer_state)") == 4
    assert source.count("clc_pipeline.producer_tail(clc_producer_state)") == 1
    assert "utils.StaticPersistentTileScheduler" not in source


def test_v13_keeps_ab_ring_generation_continuous_across_work_tiles():
    source = _source()
    assert source.count("work_idx * K_TILES + k_tile") == 2
    assert source.count("linear_k_tile % AB_STAGES") == 2
    assert source.count("linear_k_tile // AB_STAGES") == 2
    assert "total_k_tiles = cutlass.Uint32(work_idx * K_TILES)" in source
    assert source.count("for tail_offset in range(AB_STAGES):") == 1

    # Each tile advances the six-stage ring by 128 % 6 == 2 slots.  Resetting
    # stage or phase at output-tile boundaries would therefore be incorrect.
    assert 128 % 6 == 2
    assert (128 // 6) % 2 == 1


def test_v13_overlaps_two_accumulators_without_aliasing_scale_factors():
    source = _source()
    assert "SFA_TMEM_COLUMNS = (CTA_TILE[0] // 32) * MMA_KBLOCKS_PER_STAGE" in source
    assert "SFB_TMEM_COLUMNS = (CTA_TILE[1] // 32) * MMA_KBLOCKS_PER_STAGE" in source
    assert _literal_assignment("ACC_PHYSICAL_STAGES") == 2
    assert "ACC_STAGE_STRIDE = MMA_TILE[1] - SF_TMEM_COLUMNS" in source
    assert (
        "ACC_TMEM_COLUMNS = MMA_TILE[1] * ACC_PHYSICAL_STAGES - SF_TMEM_COLUMNS"
        in source
    )
    assert "cute.append(acc_shape, ACC_PHYSICAL_STAGES)" in source
    assert "ACC_STAGE_STRIDE * fake_acc.stride[0][1]" in source
    assert "sfa_ptr = cute.recast_ptr(acc_ptr + ACC_TMEM_COLUMNS" in source

    acc0 = range(0, 256)
    acc1 = range(208, 464)
    sfa = range(464, 480)
    sfb = range(480, 512)
    assert len(set(acc0) & set(acc1)) == 48
    assert not (set(acc0) | set(acc1)) & (set(sfa) | set(sfb))


def test_v13_releases_accumulator_after_tmem_fence_then_stores_directly():
    source = " ".join(_source().split())
    copy_at = source.index(
        "cute.copy( t2r, t_acc[(None, None, None, real_subtile_idx)], r_acc, )"
    )
    fence_at = source.index("cute.arch.fence_view_async_tmem_load()", copy_at)
    release_at = source.index(
        "cute.arch.mbarrier_arrive( acc_empty_mbar, acc_empty_dst_rank )",
        fence_at,
    )
    convert_at = source.index("r_c.store(r_acc.load().to(BF16))", release_at)
    store_at = source.index("cute.copy( gmem_store, r_c,", convert_at)
    assert copy_at < fence_at < release_at < convert_at < store_at
    assert "PipelineTmaStore.create(" not in source
    assert "early_release_subtile = ( cute.ceil_div(" in source
    assert "empty_phase = 1 ^ (work_idx % 2)" in source
    assert "full_phase = (work_idx // ACC_FULL_STAGES) % 2" in source
    assert "final_empty_phase = 1 ^ (work_idx % 2)" in source


def test_v13_amortizes_resource_setup_and_has_only_final_tails():
    source = _source()
    assert source.count("cute.arch.mbarrier_init(ab_full_mbar + stage, 1)") == 1
    assert source.count("tmem.allocate(TMEM_COLUMNS)") == 1
    assert source.count("tmem.relinquish_alloc_permit()") == 1
    assert source.count("tmem.free(tmem.retrieve_ptr(F32))") == 1

    kernel = _function("_dense_gemm_v13_kernel")
    while_nodes = [node for node in ast.walk(kernel) if isinstance(node, ast.While)]
    assert len(while_nodes) == 4
    for node in while_nodes:
        loop_source = ast.get_source_segment(source, node)
        assert "tmem.allocate(" not in loop_source
        assert "tmem.free(" not in loop_source
        assert "mbarrier_init(" not in loop_source


def test_v13_harness_reports_clc_grid_and_default_optimized_build():
    function = _function("dense_gemm_v13")
    assert [argument.arg for argument in function.args.args] == [
        "a_ptr",
        "b_ptr",
        "sfa_ptr",
        "sfb_ptr",
        "c_ptr",
        "max_active_clusters",
        "stream",
    ]
    assert ast.unparse(function.args.args[-2].annotation) == "cutlass.Constexpr"

    modal_source = (REPO_ROOT / "modal_dense_gemm.py").read_text(encoding="utf-8")
    iket_source = (REPO_ROOT / "cutex" / "iket_worker.py").read_text(
        encoding="utf-8"
    )
    assert '"scheduler": "CLC dynamic persistent"' in modal_source
    assert 'f"M-major, cluster swizzle {V13_CLUSTER_SWIZZLE_SIZE}"' in modal_source
    assert 'effective_compile_options = "--opt-level 1"' in modal_source
    assert 'kernel_config["logical_launch_grid"]' in modal_source
    assert 'kernel_config["resident_clusters"]' in modal_source
    assert '"manual_pipeline_v13"' in iket_source
