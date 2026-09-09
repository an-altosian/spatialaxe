"""Tests for the shared statistical helpers.

These were duplicated between the transcript QC script and the upstream helper
package. The arithmetic is deliberately unchanged from those copies, because the
thresholds they produce are compared against values recorded in historical QC
reports, so the tests pin behaviour rather than propose it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from spatialqc.stats import calculate_noise_bound, estimate_min_mols_per_cell

# ---------------------------------------------------------------------------
# calculate_noise_bound
# ---------------------------------------------------------------------------


def test_empty_input_returns_integer_zeros() -> None:
    """The integer type is load-bearing, not incidental.

    The value is serialised into the metrics JSON, where ``0`` and ``0.0`` are
    different documents, so a float would change the output contract.
    """
    lower, upper = calculate_noise_bound(pd.Series([], dtype=float))
    assert (lower, upper) == (0, 0)
    assert isinstance(lower, int) and isinstance(upper, int)


def test_bounds_bracket_the_data() -> None:
    counts = pd.Series([10, 12, 11, 13, 9, 10, 11, 12])
    lower, upper = calculate_noise_bound(counts)
    assert lower < counts.median() < upper


def test_bounds_are_positive_because_the_model_is_log_space() -> None:
    # The estimator works on log10 counts and exponentiates back, so a bound can
    # approach zero but never go negative the way a linear interval would.
    lower, _ = calculate_noise_bound(pd.Series([1, 1, 2, 1, 3]))
    assert lower > 0


def test_a_constant_series_collapses_the_interval() -> None:
    # Zero dispersion means the MAD scale is zero, so both bounds land on the value.
    lower, upper = calculate_noise_bound(pd.Series([5, 5, 5, 5]))
    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(5.0)


def test_outliers_do_not_dominate_the_scale() -> None:
    """This is why the estimator uses a median absolute deviation.

    A handful of very bright cells must not inflate the bound the way a plain
    standard deviation would.
    """
    clean = pd.Series([10] * 50 + [11] * 50)
    contaminated = pd.Series([10] * 50 + [11] * 50 + [100000] * 3)
    _, upper_clean = calculate_noise_bound(clean)
    _, upper_contaminated = calculate_noise_bound(contaminated)
    assert upper_contaminated < upper_clean * 10


def test_a_wider_quantile_widens_the_interval() -> None:
    counts = pd.Series([8, 9, 10, 11, 12, 13])
    _, upper_narrow = calculate_noise_bound(counts, quant=0.75)
    _, upper_wide = calculate_noise_bound(counts, quant=0.999)
    assert upper_wide > upper_narrow


# ---------------------------------------------------------------------------
# estimate_min_mols_per_cell
# ---------------------------------------------------------------------------


def test_never_returns_below_the_floor() -> None:
    # Sparse data would otherwise produce a threshold of 1 or 2, which admits
    # essentially every barcode as a cell.
    assert estimate_min_mols_per_cell([1, 1, 2, 2, 3], min_value=10) >= 10


def test_the_floor_is_configurable() -> None:
    assert estimate_min_mols_per_cell([1, 1, 2], min_value=42) == 42


def test_returns_an_int() -> None:
    # Used directly as a count threshold and written to the metrics JSON.
    assert isinstance(estimate_min_mols_per_cell([80, 100, 120, 140, 160]), int)


def test_a_perfectly_constant_distribution_raises() -> None:
    """Known limitation, pinned rather than fixed during the port.

    With every count identical, the histogram mode lands on the single value, so
    ``log_counts[log_counts > mode]`` is empty and ``np.quantile`` raises
    IndexError. Inherited unchanged from the pipeline script and from upstream
    ``xenium_helpers``.

    Unreachable on real data -- a sample where every cell has an identical
    molecule count does not occur -- so it is recorded as a defect rather than
    corrected here: changing the arithmetic during a mechanical port would make
    the move unverifiable against historical QC reports. See
    docs/plans/2026-09-09_PLAN_spatialqc-extraction-and-followups.md.
    """
    with pytest.raises(IndexError):
        estimate_min_mols_per_cell([100] * 100)


def test_threshold_sits_below_the_main_population() -> None:
    """The estimate is meant to cut below the bulk of real cells.

    A bimodal distribution of debris plus real cells is the realistic case.
    """
    rng = np.random.default_rng(0)
    real_cells = rng.normal(loc=300, scale=40, size=2000).clip(min=1)
    debris = rng.normal(loc=5, scale=2, size=500).clip(min=1)
    counts = np.concatenate([real_cells, debris]).astype(int)

    threshold = estimate_min_mols_per_cell(counts.tolist())
    assert threshold < np.median(real_cells)


def test_accepts_a_numpy_array() -> None:
    assert estimate_min_mols_per_cell(np.array([50, 60, 70, 80])) >= 10


def test_is_deterministic() -> None:
    counts = [10, 20, 30, 40, 50, 60] * 20
    assert estimate_min_mols_per_cell(counts) == estimate_min_mols_per_cell(counts)
