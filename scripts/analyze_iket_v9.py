"""Analyze an IKET PFTrace JSON for the dense_gemm_v9 mainloop.

Read-only diagnostic.  It reconstructs the per-k_tile timeline of the sampled
steady-state cluster -- the MMA leader wait on ``mma_wait_ab_full``, the two TMA
producers' ``tma_wait_empty`` / ``tma_issue``, and the wave/tile footprint that
follows from the full-grid warp lifetimes -- and prints the numbers needed to
reason about the ``mma_wait_ab_full`` jitter:

* per-k_tile ``mma_wait_ab_full`` duration and the TMA-completion latency it
  measured (issue_start -> full-barrier trip);
* producer lead in stages, and producer/consumer k_tile cadence;
* aggregate per-stage bytes and the implied HBM bandwidth for one resident wave;
* the tile footprint of a mid-launch wave (which M x N region is concurrently
  resident), which determines the L2-reuse / row-major-rasterization penalty.

Usage::

    python scripts/analyze_iket_v9.py <trace.json> [--stages N]

``--stages`` should match the ``AB_STAGES`` value the kernel was compiled with
(the stage index is ``k_tile % AB_STAGES``); the default 6 matches the trace
that is already in ``artifacts/``.  It does not modify any file.
"""

import argparse
import json
import statistics as S
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="path to iket *.trace.json")
    ap.add_argument("--stages", type=int, default=6)
    args = ap.parse_args()

    d = json.load(open(args.trace))
    st = d["stringTable"]
    loc = d["locationTable"]

    g = defaultdict(list)
    for r in d["launches"][0]["ranges"]:
        l = loc[r["warpLocIdxs"][0]]
        g[(tuple(l["ctaId"]), l["warpId"], st[r["rangeNameIdx"]])].append(
            (r["startTs"], r["endTs"])
        )
    for k in g:
        g[k].sort()

    mw = g[((64, 32, 0), 4, "mma_wait_ab_full")]
    mk = g[((64, 32, 0), 4, "mma_k_tile")]
    ti64 = g[((64, 32, 0), 5, "tma_issue")]
    ti65 = g[((65, 32, 0), 5, "tma_issue")]
    te64 = g[((64, 32, 0), 5, "tma_wait_empty")]
    te65 = g[((65, 32, 0), 5, "tma_wait_empty")]

    AB = args.stages
    n = len(mw)

    # ---- mma_wait_ab_full + completion latency -----------------------------
    waits = [b - a for a, b in mw]
    lat = []
    for k in range(n):
        if waits[k] > 128:
            lat.append(mw[k][1] - max(ti64[k][0], ti65[k][0]))
    print("== mma_wait_ab_full ==")
    print(f"count={n}  sum={sum(waits)}ns  "
          f"p50={S.median(waits)}ns  p95={sorted(waits)[int(n*0.95)-1]}ns  "
          f"max={max(waits)}ns  n>128ns={sum(1 for x in waits if x>128)}")
    print("== TMA completion latency for starved tiles (issue_start -> full trip) ==")
    if lat:
        print(f"n={len(lat)}  p50={S.median(lat)}ns  p95={sorted(lat)[int(len(lat)*0.95)-1]}ns  "
              f"max={max(lat)}ns  min={min(lat)}ns")

    print("== wait sum by stage index ==")
    by = defaultdict(list)
    for k in range(n):
        by[k % AB].append(waits[k])
    for s in sorted(by):
        v = by[s]
        print(f"stage={s}  sum={sum(v)}ns  n>128={sum(1 for x in v if x>128)}  max={max(v)}")

    # ---- cadence -----------------------------------------------------------
    cad = [mk[i + 1][0] - mk[i][0] for i in range(n - 1)]
    tc64 = [ti64[i + 1][0] - ti64[i][0] for i in range(n - 1)]
    tc65 = [ti65[i + 1][0] - ti65[i][0] for i in range(n - 1)]
    print("== k_tile cadence (start-to-start) ==")
    print(f"MMA   p50={S.median(cad)}ns  p95={sorted(cad)[int(len(cad)*0.95)-1]}ns  max={max(cad)}ns")
    print(f"TMA64 p50={S.median(tc64)}ns  p95={sorted(tc64)[int(len(tc64)*0.95)-1]}ns  max={max(tc64)}ns")

    # ---- producer lead in stages ------------------------------------------
    leads = []
    for k in range(n):
        issued = sum(1 for a, b in ti64 if b <= mw[k][0])
        leads.append(issued - k)
    print("== producer lead in stages (tma k_tiles issued beyond current mma k) ==")
    print({x: leads.count(x) for x in sorted(set(leads))})

    # ---- bandwidth from one cluster's cadence ------------------------------
    span = mk[-1][1] - mk[0][0]
    per_tile = span / n
    bytes_stage = 2 * (16384 + 16384 + 512 + 1024)  # pair A+B+SFA+SFB per stage
    print("== sampled cluster steady-state traffic ==")
    print(f"per-tile={per_tile:.2f}ns  pair-bytes/stage={bytes_stage}  "
          f"per-cluster-BW={bytes_stage/per_tile:.1f}GB/s")

    # ---- wave tile footprint from warp lifetimes ---------------------------
    # Report the footprint of each ~74-cluster launch window in the order the
    # scheduler launches them (consecutive physical cluster ids, row-major in
    # the 2-CTA cluster grid).  If the kernel remaps the cluster grid with a
    # Z-order curve (has ZORDER_BITS), decode the launched cid to its tile so
    # the window is measured in tile space, not physical space.
    zorder_decode = _load_zorder_decode()
    wl = d["launches"][0]["warpLifetimes"]
    cstart = defaultdict(list)
    for w in wl:
        c = tuple(loc[w["locIdx"]]["ctaId"])
        cstart[c].append(w["startTs"])
    cl_start = {}
    for c, ts in cstart.items():
        bm, bn, _ = c
        key = (bm // 2, bn)
        cl_start[key] = min(cl_start.get(key, 10 ** 30), min(ts))
    order = sorted(cl_start.items(), key=lambda kv: kv[1])
    t0 = order[0][1]

    def tile_of(cid):
        cm, cn = cid % 64, cid // 64
        return zorder_decode(cid) if zorder_decode else (cm, cn)

    print("== concurrent cluster footprint in middle of launch ==")
    for base in range(0, int((order[-1][1] - t0) / 1000), 500):
        win = [c for c, s in order if base * 1000 <= (s - t0) < (base + 70) * 1000]
        if not win:
            continue
        # launch order within the window = ascending cid (cluster grid index)
        tiles = [tile_of(k[0] + 64 * k[1]) for k in sorted(win)]
        ms = [t[0] for t in tiles]
        ns = [t[1] for t in tiles]
        print(f"t={base:4d}us  resident={len(win):3d}  "
              f"M_span={min(ms)}..{max(ms)} ({max(ms)-min(ms)})  "
              f"N_span={min(ns)}..{max(ns)} ({max(ns)-min(ns)})")


def _load_zorder_decode():
    """Return the kernel's module-level ``_zorder_decode`` if it has one."""
    import ast
    from pathlib import Path

    source_path = Path(__file__).resolve().parents[1] / "cutex" / "kernels" / "dense_gemm_v9.py"
    if not source_path.exists():
        return None
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    names = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_zorder_decode"
    }
    if "_zorder_decode" not in names:
        return None
    constants = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ZORDER_BITS" for t in node.targets)
    ]
    body = constants + [names["_zorder_decode"]]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<v9>", "exec"), ns)
    return ns["_zorder_decode"]


if __name__ == "__main__":
    main()
