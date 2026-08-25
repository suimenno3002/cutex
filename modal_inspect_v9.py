"""Host-side shape inspection for the dense_gemm_v9 implementation on Modal.

Read-only diagnostic.  It rebuilds the v9 host objects -- the two tiled MMAs,
the two cluster layouts, the four TMA paths and the four SMEM layouts -- using
the exact constants and helpers that ``dense_gemm_v9`` uses, then records every
shape / size that the discussion about ``CtaGroup.TWO`` vs ``CtaGroup.ONE`` was
based on.

CuTeDSL layout APIs can only run while a ``@cute.jit`` function is being
traced, so the construction lives inside a jit function that is *compiled but
never launched*: tracing fills a plain Python dict as a side effect, and the
real v9 device kernel is not entered.

MMA/TMA atom shapes depend on ``CtaGroup``, the tile shapes and the operand
types, not on the physical GPU, so they are identical on L4 and B300.

Run from the repo root::

    uv run modal run modal_inspect_v9.py
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import modal


APP_NAME = "cutex-inspect-v9-host-side"
app = modal.App(APP_NAME)
IMAGE = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:26.07-py3")
    .entrypoint([])
    .uv_pip_install("nvidia-cutlass-dsl[cu13]==4.7.0")
    .env({"NVIDIA_IMEX_CHANNELS": "0", "PYTHONUNBUFFERED": "1"})
    .add_local_python_source("cutex", copy=True)
)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-") or "unknown"


@app.function(
    image=IMAGE,
    gpu="L4",
    timeout=30 * 60,
)
def run_inspect() -> dict:
    import os

    # The block-scaled tcgen05 MMA only exists for SM100/SM103/SM110.  This is
    # a host-side / compile-target check, not a GPU check, and the atom SHAPES
    # we inspect depend only on CtaGroup + tile + operand types, so targeting
    # sm_103a on an L4 is fine.  Nothing is launched.
    os.environ["CUTE_DSL_ARCH"] = "sm_103a"

    import cutlass
    import cutlass.cute as cute
    import cutlass.utils.blackwell_helpers as sm100_utils
    import cutlass.utils.blockscaled_layout as blockscaled_utils
    from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
    from cutlass.cute.runtime import make_ptr

    from cutex.kernels.dense_gemm_contract import K, M, N, SF_VECTOR_SIZE
    from cutex.kernels.dense_gemm_v9 import (
        AB_STAGES,
        ACC_STAGES,
        CLUSTER_SHAPE,
        CTA_TILE,
        FP8,
        MMA_INSTRUCTION,
        MMA_TILE,
        SF8,
        SFB_MMA_TILE,
    )

    report: dict = {}

    def record(name, value):
        try:
            report[name] = int(value)
        except (TypeError, ValueError):
            report[name] = str(value)

    @cute.jit
    def inspect_host(
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
    ):
        # --- identical to the dense_gemm_v9 host side --------------------
        a = cute.make_tensor(a_ptr, cute.make_layout((M, K, 1), stride=(K, 1, M * K)))
        b = cute.make_tensor(b_ptr, cute.make_layout((N, K, 1), stride=(K, 1, N * K)))
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VECTOR_SIZE)
        )
        sfb = cute.make_tensor(
            sfb_ptr, blockscaled_utils.tile_atom_to_shape_SF(b.shape, SF_VECTOR_SIZE)
        )

        mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            FP8, FP8, OperandMajorMode.K, OperandMajorMode.K, SF8,
            SF_VECTOR_SIZE, tcgen05.CtaGroup.TWO, MMA_TILE[:2],
        )
        mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            FP8, FP8, OperandMajorMode.K, OperandMajorMode.K, SF8,
            SF_VECTOR_SIZE, tcgen05.CtaGroup.ONE, SFB_MMA_TILE[:2],
        )

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(CLUSTER_SHAPE), (mma.thr_id.shape,)
        )
        cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout(CLUSTER_SHAPE), (mma_sfb.thr_id.shape,)
        )

        s_layout_a = sm100_utils.make_smem_layout_a(mma, MMA_TILE, FP8, AB_STAGES)
        s_layout_b = sm100_utils.make_smem_layout_b(mma, MMA_TILE, FP8, AB_STAGES)
        s_layout_sfa = blockscaled_utils.make_smem_layout_sfa(
            mma, MMA_TILE, SF_VECTOR_SIZE, AB_STAGES
        )
        s_layout_sfb = blockscaled_utils.make_smem_layout_sfb(
            mma, MMA_TILE, SF_VECTOR_SIZE, AB_STAGES
        )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE[:2], mma.thr_id)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(CLUSTER_SHAPE[:2], mma.thr_id)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE[:2], mma.thr_id)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            CLUSTER_SHAPE[:2], mma.thr_id
        )

        tma_a, g_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, cute.slice_(s_layout_a, (None, None, None, 0)),
            MMA_TILE, mma, cluster_layout_vmnk.shape,
        )
        tma_b, g_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, cute.slice_(s_layout_b, (None, None, None, 0)),
            MMA_TILE, mma, cluster_layout_vmnk.shape,
        )
        tma_sfa, g_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa, cute.slice_(s_layout_sfa, (None, None, None, 0)),
            MMA_TILE, mma, cluster_layout_vmnk.shape, internal_type=cutlass.Int16,
        )
        tma_sfb, g_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb, cute.slice_(s_layout_sfb, (None, None, None, 0)),
            SFB_MMA_TILE, mma_sfb, cluster_layout_sfb_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        s_a0 = cute.size_in_bytes(FP8, cute.slice_(s_layout_a, (None, None, None, 0)))
        s_b0 = cute.size_in_bytes(FP8, cute.slice_(s_layout_b, (None, None, None, 0)))
        s_sfa0 = cute.size_in_bytes(
            SF8, cute.slice_(s_layout_sfa, (None, None, None, 0))
        )
        s_sfb0 = cute.size_in_bytes(
            SF8, cute.slice_(s_layout_sfb, (None, None, None, 0))
        )

        # --- record everything (plain Python side effect during tracing) --
        record("mma_thr_id", mma.thr_id)
        record("mma_sfb_thr_id", mma_sfb.thr_id)
        record("mma_thr_id_size", cute.size(mma.thr_id.shape))
        record("mma_sfb_thr_id_size", cute.size(mma_sfb.thr_id.shape))
        record("cluster_layout_vmnk", cluster_layout_vmnk)
        record("cluster_layout_sfb_vmnk", cluster_layout_sfb_vmnk)
        record("tma_a_atom", tma_a)
        record("tma_b_atom", tma_b)
        record("tma_sfa_atom", tma_sfa)
        record("tma_sfb_atom", tma_sfb)
        record("tma_a_op", a_op)
        record("tma_b_op", b_op)
        record("tma_sfa_op", sfa_op)
        record("tma_sfb_op", sfb_op)
        record("g_a_layout", g_a.layout)
        record("g_b_layout", g_b.layout)
        record("g_sfa_layout", g_sfa.layout)
        record("g_sfb_layout", g_sfb.layout)
        record("sfa_gmem_layout", sfa.layout)
        record("sfb_gmem_layout", sfb.layout)
        record("s_layout_a", s_layout_a)
        record("s_layout_b", s_layout_b)
        record("s_layout_sfa", s_layout_sfa)
        record("s_layout_sfb", s_layout_sfb)
        record("stage_bytes_A", s_a0)
        record("stage_bytes_B", s_b0)
        record("stage_bytes_SFA", s_sfa0)
        record("stage_bytes_SFB", s_sfb0)
        record("tma_bytes", (s_a0 + s_b0 + s_sfa0 + s_sfb0) * cute.size(mma.thr_id.shape))

    a_ptr = make_ptr(FP8, 0, cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(FP8, 0, cute.AddressSpace.gmem, assumed_align=16)
    sfa_ptr = make_ptr(SF8, 0, cute.AddressSpace.gmem, assumed_align=32)
    sfb_ptr = make_ptr(SF8, 0, cute.AddressSpace.gmem, assumed_align=32)

    # Compile (trace) only; nothing is launched and no v9 device code runs.
    cute.compile(inspect_host, a_ptr, b_ptr, sfa_ptr, sfb_ptr)

    report["constants"] = {
        "MMA_INSTRUCTION": list(MMA_INSTRUCTION),
        "MMA_TILE": list(MMA_TILE),
        "CTA_TILE": list(CTA_TILE),
        "SFB_MMA_TILE": list(SFB_MMA_TILE),
        "CLUSTER_SHAPE": list(CLUSTER_SHAPE),
        "AB_STAGES": AB_STAGES,
        "ACC_STAGES": ACC_STAGES,
    }
    report["_meta"] = {
        "note": "host-side atom/layout shapes captured during jit tracing; "
        "no kernel was launched",
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


@app.local_entrypoint()
def main():
    result = run_inspect.remote()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out_dir = Path(__file__).parent / "artifacts" / f"v9-host-inspect-{_safe_slug(run_id)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "inspect.json"
    out_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    print(f"local_artifact: {out_path.resolve()}")
