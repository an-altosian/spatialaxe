"""Transcript quality control for Xenium decoded transcripts.

Grades the decoded transcript table and the cell-feature matrix: per-cell and
per-gene count distributions, negative-control probe rates, and
transcript-to-cell assignment statistics.

This analysis streams Parquet through pyarrow and has no CUDA or Numba code
path, so it takes no ``device`` argument; GPU/CPU selection is specific to
:mod:`spatialqc.image`.

Exports resolve lazily (PEP 562) so that importing the streaming helpers --
``from spatialqc.transcript import stream`` -- does not pull in scanpy, which
only the full analysis needs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["TranscriptQCOptions", "run_transcript_qc"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spatialqc.transcript.qc import TranscriptQCOptions, run_transcript_qc


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module("spatialqc.transcript.qc"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
