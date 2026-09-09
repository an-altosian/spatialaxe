"""spatialqc -- quality-control analytics for 10x Genomics Xenium runs.

Two analyses live here, each usable as a library call or a console script:

* **Image QC** (:func:`run_image_qc`) grades the morphology images of a Xenium
  bundle: per-tile focus and contrast maps, region-of-interest blur
  classification, stain intensity assessment, signal-to-noise ratio, and
  optional per-cell texture metrics.
* **Transcript QC** (:func:`run_transcript_qc`) grades the decoded transcript
  table and cell-feature matrix: per-cell and per-gene count distributions,
  negative-control probe rates, and assignment statistics.

Both accept a ``device`` selector so the numerical kernels run on either a CUDA
GPU or the CPU; see :mod:`spatialqc.backend`.

Typical library use::

    from pathlib import Path
    from spatialqc import run_image_qc

    result = run_image_qc(
        xenium_bundle_dir=Path("/data/output-XETG00001"),
        outdir=Path("qc/image"),
        sample_id="sample_a",
        device="auto",
    )

Importing this package is cheap: the heavy submodules (scikit-image, scanpy) are
loaded on first attribute access via PEP 562, so ``import spatialqc`` does not
pay for a stack the caller may not use.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

#: Authoritative version for the distribution. ``pyproject.toml`` reads this
#: attribute via ``[tool.setuptools.dynamic]``, so it is declared exactly once.
__version__ = "0.1.0"

__all__ = [
    "Backend",
    "Device",
    "__version__",
    "backend",
    "get_backend",
    "run_image_qc",
    "run_transcript_qc",
]

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from spatialqc.backend import Backend, Device, get_backend
    from spatialqc.image import run_image_qc
    from spatialqc.transcript import run_transcript_qc


# Map each lazily-exported name to the submodule that defines it. Keeping the
# heavy imports out of module scope is what makes `import spatialqc` fast and
# what lets a consumer install only the extra they need: asking for
# `run_transcript_qc` never imports the imaging stack.
_LAZY_EXPORTS: dict[str, str] = {
    "Backend": "spatialqc.backend",
    "Device": "spatialqc.backend",
    "get_backend": "spatialqc.backend",
    "run_image_qc": "spatialqc.image",
    "run_transcript_qc": "spatialqc.transcript",
}


def __getattr__(name: str) -> Any:
    """Resolve public names on first access (PEP 562)."""
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    # Cache on the module so subsequent lookups skip __getattr__ entirely.
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
