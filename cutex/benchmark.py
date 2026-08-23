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

