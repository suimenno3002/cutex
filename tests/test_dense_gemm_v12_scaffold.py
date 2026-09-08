import ast
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "cutex" / "kernels" / "dense_gemm_v12.py"
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


def test_v12_preserves_v11_math_tile_cluster_and_warp_roles():
    assert _literal_assignment("MMA_INSTRUCTION") == (256, 256, 32)
    assert _literal_assignment("MMA_TILE") == (256, 256, 128)
    assert _literal_assignment("CTA_TILE") == (128, 256, 128)
    assert _literal_assignment("SFB_MMA_TILE") == (128, 256, 128)
    assert _literal_assignment("CLUSTER_SHAPE") == (2, 1, 1)
    assert _literal_assignment("CLUSTER_SWIZZLE_SIZE") == 8
    assert _literal_assignment("EPILOGUE_WARPS") == (0, 1, 2, 3)
    source = _source()
    assert "AB_STAGES, ACC_STAGES = 6, 1" in source
    assert "MMA_WARP = 4" in source
    assert "TMA_WARP = 5" in source
    assert "THREADS = 6 * 32" in source


def test_v12_uses_native_static_persistent_scheduling_and_one_wave_grid():
    source = _source()
    assert "utils.PersistentTileSchedulerParams(" in source
    assert "utils.StaticPersistentTileScheduler.get_grid_shape(" in source
    assert "utils.StaticPersistentTileScheduler.create(" in source
    assert "CLUSTER_SWIZZLE_SIZE," in source
    assert "max_active_clusters," in source
    assert "grid=grid" in source
    assert source.count("while work_tile.is_valid_tile:") == 3
    assert source.count("tile_sched.advance_to_next_work()") == 3
    assert "grid=(M // CTA_TILE[0], N // CTA_TILE[1], 1)" not in source

    # Static scheduler rank r visits r, r+74, ... in the 4096-cluster logical
    # problem.  Together the 74 resident clusters cover every tile once.
    sequences = [list(range(rank, 64 * 64, 74)) for rank in range(74)]
    flattened = [rank for sequence in sequences for rank in sequence]
    assert sorted(flattened) == list(range(64 * 64))
    assert {len(sequence) for sequence in sequences} == {55, 56}


def test_v12_keeps_ab_ring_generation_continuous_across_work_tiles():
    source = _source()
    assert source.count("linear_k_tile = work_idx * K_TILES + k_tile") == 2
    assert source.count("stage = linear_k_tile % AB_STAGES") == 2
    assert "generation = (linear_k_tile // AB_STAGES) % 2" in source
    assert "full_phase = (linear_k_tile // AB_STAGES) % 2" in source
    assert "total_k_tiles = tile_sched.num_tiles_executed * K_TILES" in source
    assert source.count("for tail_offset in range(AB_STAGES):") == 1

    # K_TILES=128 is not divisible by six, so resetting phase at every work
    # tile would be wrong.  The global count advances the next tile by two ring
    # slots and flips generation according to the true linear position.
    assert 128 % 6 == 2
    assert (128 // 6) % 2 == 1


def test_v12_overlaps_two_accumulator_views_without_aliasing_scale_factors():
    source = _source()
    assert _literal_assignment("SFA_TMEM_COLUMNS") == 16
    assert _literal_assignment("SFB_TMEM_COLUMNS") == 32
    assert _literal_assignment("ACC_PHYSICAL_STAGES") == 2
    assert "ACC_STAGE_STRIDE = MMA_TILE[1] - SF_TMEM_COLUMNS" in source
    assert (
        "ACC_TMEM_COLUMNS = MMA_TILE[1] * ACC_PHYSICAL_STAGES - SF_TMEM_COLUMNS"
        in source
    )
    assert "cute.append(acc_shape, ACC_PHYSICAL_STAGES)" in source
    assert "ACC_STAGE_STRIDE * fake_acc.stride[0][1]" in source
    assert "sfa_ptr = cute.recast_ptr(acc_ptr + ACC_TMEM_COLUMNS" in source
    assert "acc_stage = work_idx % ACC_PHYSICAL_STAGES" in source

    acc0 = range(0, 256)
    acc1 = range(208, 464)
    sfa = range(464, 480)
    sfb = range(480, 512)
    assert len(set(acc0) & set(acc1)) == 48
    assert not (set(acc0) | set(acc1)) & (set(sfa) | set(sfb))


def test_v12_releases_accumulator_only_after_tmem_copy_fence():
    source = _source()
    copy_at = source.index("cute.copy(t2r, t_acc, r_acc)")
    fence_at = source.index("cute.arch.fence_view_async_tmem_load()", copy_at)
    release_at = source.index(
        "cute.arch.mbarrier_arrive(acc_empty_mbar, acc_empty_dst_rank)", fence_at
    )
    convert_at = source.index("r_out.store(r_acc.load().to(BF16))", release_at)
    store_at = source.index("cute.copy(gmem_store, r_out, out)", convert_at)
    assert copy_at < fence_at < release_at < convert_at < store_at
    assert "empty_phase = 1 ^ (work_idx % 2)" in source
    assert "full_phase = work_idx % 2" in source
    assert "final_empty_phase = 1 ^ (tile_sched.num_tiles_executed % 2)" in source


def test_v12_amortizes_barrier_and_tmem_prologue_and_has_one_final_epilogue():
    source = _source()
    assert source.count("cute.arch.mbarrier_init(ab_full_mbar + stage, 1)") == 1
    assert source.count("tmem.allocate(TMEM_COLUMNS)") == 1
    assert source.count("tmem.relinquish_alloc_permit()") == 1
    assert source.count("tmem.free(tmem.retrieve_ptr(F32))") == 1

    kernel = _function("_dense_gemm_v12_kernel")
    while_nodes = [node for node in ast.walk(kernel) if isinstance(node, ast.While)]
    assert len(while_nodes) == 3
    for node in while_nodes:
        loop_source = ast.get_source_segment(source, node)
        assert "tmem.allocate(" not in loop_source
        assert "tmem.free(" not in loop_source
        assert "mbarrier_init(" not in loop_source


def test_v12_uses_a_compile_time_runtime_occupancy_limit():
    function = _function("dense_gemm_v12")
    assert [argument.arg for argument in function.args.args] == [
        "a_ptr",
        "b_ptr",
        "sfa_ptr",
        "sfb_ptr",
        "c_ptr",
        "max_active_clusters",
        "stream",
    ]
    parameter = function.args.args[-2]
    assert ast.unparse(parameter.annotation) == "cutlass.Constexpr"

    modal_source = (REPO_ROOT / "modal_dense_gemm.py").read_text(encoding="utf-8")
    iket_source = (REPO_ROOT / "cutex" / "iket_worker.py").read_text(
        encoding="utf-8"
    )
    for harness_source in (modal_source, iket_source):
        assert "cutlass_utils.HardwareInfo().get_max_active_clusters(" in harness_source
        assert "max_active_clusters," in harness_source
    assert 'implementation == "manual_pipeline_v12"' in modal_source
    assert '"manual_pipeline_v12"' in iket_source
