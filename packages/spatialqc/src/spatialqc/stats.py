"""Statistical helpers shared by the QC analyses.

These are ported byte-for-byte in behaviour from the pipeline scripts, where
they were duplicated between ``bin/transcript_qc_processing.py`` and the
upstream ``xenium_helpers.utils``. Only the surface was modernised (typing,
naming of locals); the arithmetic is unchanged, because the thresholds they
produce are compared against values recorded in historical QC reports.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

__all__ = ["calculate_noise_bound", "estimate_min_mols_per_cell"]


def calculate_noise_bound(
    n_molecules_non_gene_prefix: pd.Series,
    quant: float = 0.99,
) -> tuple[float, float]:
    """Bound the per-cell background level from non-gene (control) molecules.

    Works in log10 space with a median-absolute-deviation scale estimate, so a
    minority of very bright cells cannot inflate the bound the way a plain
    standard deviation would.

    Args:
        n_molecules_non_gene_prefix: Per-cell counts of molecules whose feature
            name matches a non-gene prefix (negative controls, blanks).
        quant: Two-sided normal quantile for the interval, e.g. ``0.99``.

    Returns:
        ``(lower, upper)`` bounds in linear counts.

        Returns the integers ``(0, 0)`` for an empty input. That exact type is
        preserved from the original: the value is serialised into the metrics
        JSON, where ``0`` and ``0.0`` are different documents.
    """
    from scipy.stats import median_abs_deviation, norm

    if n_molecules_non_gene_prefix.empty:
        return 0, 0

    quant_val = norm.ppf(quant)
    log_counts = np.log10(n_molecules_non_gene_prefix.values)
    scale = median_abs_deviation(log_counts, scale="normal")
    lower = np.mean(log_counts) - quant_val * scale
    upper = np.mean(log_counts) + quant_val * scale
    return 10**lower, 10**upper


def estimate_min_mols_per_cell(
    n_mols_per_cell: Sequence[int] | np.ndarray,
    min_value: int = 10,
) -> int:
    """Choose a per-cell molecule-count floor from the count distribution.

    Locates the mode of the log10 count histogram -- for Xenium data this is the
    bulk of real cells -- and steps back from it by the distance to the 99th
    percentile of the upper tail, giving a cut-off below the main population
    rather than at an arbitrary constant.

    Args:
        n_mols_per_cell: Per-cell molecule counts.
        min_value: Absolute floor; the estimate never returns less than this.

    Returns:
        The molecule-count threshold, at least *min_value*.
    """
    log_counts = np.log10(np.asarray(n_mols_per_cell) + 1)
    histogram, edges = np.histogram(log_counts, bins=100)
    mode = edges[histogram.argmax()]
    spread = np.quantile(log_counts[log_counts > mode], 0.99) - mode
    return max(min_value, int(round(10 ** (mode - spread))))
