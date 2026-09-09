"""Exception hierarchy for :mod:`spatialqc`.

A single base class lets a caller distinguish "the QC analysis rejected this
input" from an unexpected crash. The Nextflow modules rely on that distinction:
a :class:`SpatialQCError` exits 1 and still writes a status JSON so the report
renders a QC-FAILED banner, whereas an unexpected traceback propagates so the
retry ``errorStrategy`` can fire.
"""

from __future__ import annotations

__all__ = [
    "BundleError",
    "InsufficientDataError",
    "SpatialQCError",
    "ThresholdConfigError",
]


class SpatialQCError(Exception):
    """Base class for every error raised deliberately by spatialqc."""


class BundleError(SpatialQCError):
    """A Xenium bundle is missing a required file or is not readable."""


class InsufficientDataError(SpatialQCError):
    """The input is valid but too sparse to produce a QC verdict.

    Raised, for example, when no tissue tile clears the intensity gate on a very
    dim sample. This is an expected outcome, not a bug.
    """


class ThresholdConfigError(SpatialQCError):
    """A threshold YAML is malformed or names an unknown channel/metric."""
