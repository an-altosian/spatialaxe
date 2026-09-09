"""Tests for the package's public API contract.

These guard the packaging properties that are easy to regress silently: the
version is single-sourced, the advertised names resolve, and importing the
package does not drag in the heavy optional stacks.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

import spatialqc


def test_version_is_a_non_placeholder_string() -> None:
    assert isinstance(spatialqc.__version__, str)
    assert spatialqc.__version__ not in {"", "0.0.0", "0.0.0.dev0"}


def test_installed_metadata_matches_the_module_attribute() -> None:
    """pyproject reads the version from this attribute, so the two must agree.

    A mismatch means the distribution was built from a different tree than the
    one being imported.
    """
    from importlib.metadata import version

    assert version("spatialqc") == spatialqc.__version__


@pytest.mark.parametrize("name", ["get_backend", "Backend", "Device"])
def test_backend_names_resolve_lazily(name: str) -> None:
    assert getattr(spatialqc, name) is not None


def test_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError, match="has no attribute"):
        spatialqc.no_such_symbol  # noqa: B018


def test_dir_reports_the_public_api() -> None:
    assert set(spatialqc.__dir__()) == set(spatialqc.__all__)


def test_lazy_access_is_cached() -> None:
    first = spatialqc.get_backend
    assert spatialqc.get_backend is first
    # After resolution the name lives in the module globals, bypassing __getattr__.
    assert "get_backend" in vars(spatialqc)


def test_importing_the_package_does_not_import_the_heavy_stacks() -> None:
    """`import spatialqc` must stay cheap.

    Run in a subprocess because this test session has almost certainly imported
    scipy and friends already, which would make an in-process check meaningless.
    """
    code = (
        "import sys, spatialqc;"
        "heavy = [m for m in ('scanpy','skimage','sklearn','seaborn','tifffile','zarr')"
        " if m in sys.modules];"
        "print(','.join(heavy))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "", f"eagerly imported: {out.stdout.strip()}"


def test_backend_submodule_is_importable_without_extras() -> None:
    # backend imports scipy lazily inside NumpyBackend.__init__, so the module
    # itself must import even where the `image` extra is absent.
    import spatialqc.backend as backend_module

    assert hasattr(backend_module, "get_backend")


def test_exceptions_share_one_base() -> None:
    from spatialqc.exceptions import (
        BundleError,
        InsufficientDataError,
        SpatialQCError,
        ThresholdConfigError,
    )

    for exc in (BundleError, InsufficientDataError, ThresholdConfigError):
        assert issubclass(exc, SpatialQCError)
