"""Shared fixtures for the spatialqc test suite.

There is deliberately **no** ``sys.path`` manipulation here. The package uses a
``src/`` layout, so the tests import the *installed* distribution
(``pip install -e '.[dev,all]'``). That is what makes the suite safe under
``pytest-xdist``: the previous pipeline-script arrangement inserted paths in
``conftest.py``, which made collection order load-bearing and let a module pass
only when some sibling had already imported the real directory.

All fixtures generate synthetic data with a fixed seed: no network, no bundle on
disk, no reference files.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

#: Fixed seed so every synthetic array is reproducible across runs and workers.
SEED = 42


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded random generator."""
    return np.random.default_rng(SEED)


@pytest.fixture
def noise_tile(rng: np.random.Generator) -> NDArray[np.float32]:
    """A small featureless float32 tile."""
    return rng.random((64, 64), dtype=np.float32)


@pytest.fixture
def structured_tile() -> NDArray[np.float32]:
    """A tile with real spatial structure.

    Filter kernels (uniform, Laplacian, Laplacian-of-Gaussian) are only
    meaningfully exercised by an image with edges and a gradient; pure noise
    hides sign and orientation errors.
    """
    yy, xx = np.mgrid[0:64, 0:64]
    tile = np.zeros((64, 64), dtype=np.float32)
    tile += (xx / 63.0).astype(np.float32)  # smooth horizontal ramp
    tile[16:48, 16:48] += 2.0  # a hard-edged bright square
    tile[30:34, :] += 0.5  # a thin horizontal bar
    return tile


@pytest.fixture
def uint16_tile(rng: np.random.Generator) -> NDArray[np.uint16]:
    """A 16-bit tile shaped like real morphology data (bright blob on dim background)."""
    tile = rng.integers(80, 160, size=(64, 64), dtype=np.uint16)
    tile[20:44, 20:44] = rng.integers(3000, 6000, size=(24, 24), dtype=np.uint16)
    return tile
