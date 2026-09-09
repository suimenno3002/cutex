from __future__ import annotations

import pytest

from cutex.benchmark import paired_comparison, stats_from_samples


def test_paired_comparison_preserves_pairing_and_direction() -> None:
    custom = stats_from_samples([9.0, 12.0, 10.0])
    reference = stats_from_samples([10.0, 10.0, 11.0])

    paired = paired_comparison(custom, reference)

    assert paired["sample_pairs"] == 3
    assert paired["delta_us"]["samples"] == [-1.0, 2.0, -1.0]
    assert paired["delta_us"]["median"] == -1.0
    assert paired["efficiency_pct"]["median"] == pytest.approx(110.0)
    assert paired["first_faster_samples"] == 2
    assert paired["first_faster_pct"] == pytest.approx(200.0 / 3.0)


def test_paired_comparison_rejects_mismatched_counts() -> None:
    one = stats_from_samples([1.0])
    two = stats_from_samples([1.0, 2.0])

    with pytest.raises(ValueError, match="counts"):
        paired_comparison(one, two)
