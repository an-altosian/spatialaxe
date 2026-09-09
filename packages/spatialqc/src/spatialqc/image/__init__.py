"""Image quality control for Xenium morphology images.

Grades the morphology image stack of a Xenium bundle: per-tile focus and
contrast maps, region-of-interest blur classification via a Gaussian mixture,
stain intensity assessment, signal-to-noise ratio, and -- when cell
segmentation is present -- per-cell texture metrics mapped back from the tile
grid.

The numerical kernels run on either a CUDA GPU or the CPU; see
:mod:`spatialqc.backend` and the ``device`` argument of :func:`run_image_qc`.

Exports resolve lazily (PEP 562) so that importing the SNR helpers --
``from spatialqc.image import snr`` -- does not pull in the full imaging stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__all__ = ["run_image_qc"]

if TYPE_CHECKING:  # pragma: no cover - typing only
    from spatialqc.image.qc import run_image_qc


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module("spatialqc.image.qc"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
