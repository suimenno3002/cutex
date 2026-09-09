"""GPU benchmarking helpers based on CUDA timestamps."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class BenchmarkStats:
    samples_us: tuple[float, ...]
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    p95_us: float

    def to_dict(self, *, include_samples: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_samples:
            payload.pop("samples_us", None)
        else:
            payload["samples_us"] = list(self.samples_us)
        return payload


def stats_from_samples(samples_us: list[float] | tuple[float, ...]) -> BenchmarkStats:
    if not samples_us:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(float(sample) for sample in samples_us)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return BenchmarkStats(
        samples_us=tuple(float(sample) for sample in samples_us),
        mean_us=statistics.fmean(ordered),
        median_us=statistics.median(ordered),
        min_us=ordered[0],
        max_us=ordered[-1],
        p95_us=ordered[p95_index],
    )


def cuda_benchmark(
    fn: Callable[..., Any],
    *args: Any,
    warmup: int = 5,
    rep: int = 30,
    stream=None,
    **kwargs: Any,
) -> BenchmarkStats:
    """Measure one asynchronous CUDA launch per sample.

    CUDA events timestamp work on the GPU timeline, so Python launch overhead is
    excluded. All events are queued before the final synchronization.
    """

    if warmup < 0 or rep < 1:
        raise ValueError("warmup must be >= 0 and rep must be >= 1")

    import torch

    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    pairs = []
    for _ in range(rep):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        if stream is None:
            start.record()
            fn(*args, **kwargs)
            end.record()
        else:
            start.record(stream)
            fn(*args, **kwargs)
            end.record(stream)
        pairs.append((start, end))

    pairs[-1][1].synchronize()
    samples_us = [start.elapsed_time(end) * 1_000.0 for start, end in pairs]
    return stats_from_samples(samples_us)


def cuda_benchmark_pair(
    first_fn: Callable[[], Any],
    second_fn: Callable[[], Any],
    *,
    warmup: int = 5,
    rep: int = 30,
    stream=None,
) -> tuple[BenchmarkStats, BenchmarkStats]:
    """Measure two asynchronous launches at matched thermal time points.

    Each sample pair contains one launch of each callable.  Their order flips
    on every pair so neither implementation systematically runs earlier in the
    benchmark or immediately after the same predecessor.  CUDA events still
    delimit each individual launch, excluding Python dispatch overhead.
    """

    if warmup < 0 or rep < 1:
        raise ValueError("warmup must be >= 0 and rep must be >= 1")

    import torch

    for warmup_idx in range(warmup):
        if warmup_idx % 2 == 0:
            first_fn()
            second_fn()
        else:
            second_fn()
            first_fn()
    torch.cuda.synchronize()

    event_pairs: list[list[tuple[Any, Any]]] = [[], []]
    for sample_idx in range(rep):
        order = (0, 1) if sample_idx % 2 == 0 else (1, 0)
        for fn_idx in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if stream is None:
                start.record()
                (first_fn if fn_idx == 0 else second_fn)()
                end.record()
            else:
                start.record(stream)
                (first_fn if fn_idx == 0 else second_fn)()
                end.record(stream)
            event_pairs[fn_idx].append((start, end))

    event_pairs[order[-1]][-1][1].synchronize()
    first_samples = [
        start.elapsed_time(end) * 1_000.0 for start, end in event_pairs[0]
    ]
    second_samples = [
        start.elapsed_time(end) * 1_000.0 for start, end in event_pairs[1]
    ]
    return stats_from_samples(first_samples), stats_from_samples(second_samples)


def paired_comparison(
    first: BenchmarkStats,
    second: BenchmarkStats,
) -> dict[str, Any]:
    """Summarize aligned sample pairs as latency deltas and efficiency."""

    if len(first.samples_us) != len(second.samples_us):
        raise ValueError("paired sample counts must match")
    if not first.samples_us:
        raise ValueError("at least one sample pair is required")
    if any(sample <= 0 for sample in first.samples_us):
        raise ValueError("first samples must be positive")

    deltas = [
        first_sample - second_sample
        for first_sample, second_sample in zip(
            first.samples_us, second.samples_us
        )
    ]
    efficiencies = [
        100.0 * second_sample / first_sample
        for first_sample, second_sample in zip(
            first.samples_us, second.samples_us
        )
    ]

    def metric_payload(samples: list[float]) -> dict[str, Any]:
        stats = stats_from_samples(samples)
        return {
            "samples": list(stats.samples_us),
            "mean": stats.mean_us,
            "median": stats.median_us,
            "min": stats.min_us,
            "max": stats.max_us,
            "p95": stats.p95_us,
        }

    first_faster_samples = sum(delta < 0 for delta in deltas)
    return {
        "sample_pairs": len(deltas),
        "delta_us": metric_payload(deltas),
        "efficiency_pct": metric_payload(efficiencies),
        "first_faster_samples": first_faster_samples,
        "first_faster_pct": 100.0 * first_faster_samples / len(deltas),
    }
