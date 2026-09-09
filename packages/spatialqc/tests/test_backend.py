"""Tests for the CPU/GPU backend dispatch layer.

The GPU-specific tests are marked ``gpu`` and skip when CuPy or a CUDA device is
absent, so the suite is meaningful on a CPU-only host and complete on a GPU node.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from spatialqc.backend import (
    HAS_CUPY,
    Backend,
    CupyBackend,
    NumpyBackend,
    available_gpu_ids,
    get_backend,
    gpu_is_available,
    sanitize,
)

requires_gpu = pytest.mark.skipif(not HAS_CUPY, reason="CuPy is not installed")


# ---------------------------------------------------------------------------
# Selector semantics
# ---------------------------------------------------------------------------


def test_cpu_selector_returns_numpy_backend() -> None:
    backend = get_backend("cpu")
    assert isinstance(backend, NumpyBackend)
    assert backend.name == "cpu"
    assert backend.xp is np


def test_cpu_backend_satisfies_the_backend_protocol() -> None:
    # A runtime_checkable Protocol only verifies attribute presence, which is
    # exactly the contract the numerical kernels depend on.
    assert isinstance(get_backend("cpu"), Backend)


def test_unknown_selector_raises_value_error() -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        get_backend("tpu")  # type: ignore[arg-type]


def test_auto_selector_matches_gpu_availability() -> None:
    backend = get_backend("auto")
    expected = "gpu" if gpu_is_available() else "cpu"
    assert backend.name == expected, "auto must follow actual device availability"


@pytest.mark.skipif(gpu_is_available(), reason="a CUDA device is present")
def test_explicit_gpu_raises_when_no_device_rather_than_falling_back() -> None:
    """The central safety property: no silent CPU fallback.

    A run that quietly drops to CPU returns a correct-looking result at roughly
    50x the cost, on a node requested precisely because it had a GPU.
    """
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        get_backend("gpu")


def test_available_gpu_ids_respects_the_cap() -> None:
    # max_gpus exists because an AWS Batch accelerator request does not restrict
    # what CUDA can see: a task asking for one GPU on a four-GPU instance would
    # otherwise use all four.
    assert available_gpu_ids(max_gpus=0) == []
    assert len(available_gpu_ids(max_gpus=1)) <= 1


def test_gpu_ids_empty_without_cupy() -> None:
    if not HAS_CUPY:
        assert available_gpu_ids() == []
        assert gpu_is_available() is False


# ---------------------------------------------------------------------------
# CPU primitives
# ---------------------------------------------------------------------------


def test_to_device_is_identity_on_cpu(structured_tile: NDArray[np.float32]) -> None:
    backend = get_backend("cpu")
    # Identity, not a copy: the CPU path must not pay a duplication cost per tile.
    assert backend.to_device(structured_tile) is structured_tile


def test_to_numpy_returns_float32(uint16_tile: NDArray[np.uint16]) -> None:
    backend = get_backend("cpu")
    out = backend.to_numpy(uint16_tile)
    assert out.dtype == np.float32


def test_uniform_filter_preserves_shape_and_smooths(
    structured_tile: NDArray[np.float32],
) -> None:
    backend = get_backend("cpu")
    filtered = backend.uniform_filter(structured_tile, 5)
    assert filtered.shape == structured_tile.shape
    # A box filter must reduce variance on a tile that has a hard edge.
    assert filtered.var() < structured_tile.var()


def test_uniform_filter_of_a_constant_field_is_that_constant() -> None:
    backend = get_backend("cpu")
    flat = np.full((32, 32), 7.5, dtype=np.float32)
    assert np.allclose(backend.uniform_filter(flat, 3), 7.5)


def test_laplace_is_zero_on_a_constant_field() -> None:
    backend = get_backend("cpu")
    flat = np.full((32, 32), 3.0, dtype=np.float32)
    assert np.allclose(backend.laplace(flat), 0.0)


def test_laplace_responds_at_edges(structured_tile: NDArray[np.float32]) -> None:
    backend = get_backend("cpu")
    response = np.abs(backend.laplace(structured_tile))
    # The bright square's boundary must produce a stronger response than its flat interior.
    assert response[16, 32] > response[30, 20]


def test_gaussian_laplace_suppresses_noise_relative_to_plain_laplace(
    noise_tile: NDArray[np.float32],
) -> None:
    backend = get_backend("cpu")
    plain = np.abs(backend.laplace(noise_tile)).mean()
    smoothed = np.abs(backend.gaussian_laplace(noise_tile, 2.0)).mean()
    # That pre-smoothing is the whole point of the lap_sigma parameter.
    assert smoothed < plain


def test_synchronize_is_a_noop_on_cpu() -> None:
    assert get_backend("cpu").synchronize() is None


def test_repr_identifies_the_backend() -> None:
    assert repr(get_backend("cpu")) == "NumpyBackend()"


# ---------------------------------------------------------------------------
# sanitize
# ---------------------------------------------------------------------------


def test_sanitize_replaces_non_finite_values_with_zero() -> None:
    arr = np.array([1.0, np.nan, np.inf, -np.inf, -2.5], dtype=np.float32)
    assert sanitize(arr).tolist() == [1.0, 0.0, 0.0, 0.0, -2.5]


def test_sanitize_mutates_in_place() -> None:
    arr = np.array([np.nan], dtype=np.float32)
    assert sanitize(arr) is arr


# ---------------------------------------------------------------------------
# GPU path and cross-backend parity
# ---------------------------------------------------------------------------


@requires_gpu
@pytest.mark.gpu
def test_gpu_backend_round_trips_arrays(structured_tile: NDArray[np.float32]) -> None:
    backend = CupyBackend()
    restored = backend.to_numpy(backend.to_device(structured_tile))
    assert np.allclose(restored, structured_tile, rtol=1e-6, atol=1e-6)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("size", [3, 5, 9])
def test_uniform_filter_parity_cpu_vs_gpu(structured_tile: NDArray[np.float32], size: int) -> None:
    cpu, gpu = NumpyBackend(), CupyBackend()
    expected = cpu.to_numpy(cpu.uniform_filter(cpu.to_device(structured_tile), size))
    actual = gpu.to_numpy(gpu.uniform_filter(gpu.to_device(structured_tile), size))
    assert np.allclose(expected, actual, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.gpu
def test_laplace_parity_cpu_vs_gpu(structured_tile: NDArray[np.float32]) -> None:
    cpu, gpu = NumpyBackend(), CupyBackend()
    expected = cpu.to_numpy(cpu.laplace(cpu.to_device(structured_tile)))
    actual = gpu.to_numpy(gpu.laplace(gpu.to_device(structured_tile)))
    assert np.allclose(expected, actual, rtol=1e-5, atol=1e-5)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("sigma", [1.0, 2.0])
def test_gaussian_laplace_parity_cpu_vs_gpu(
    structured_tile: NDArray[np.float32], sigma: float
) -> None:
    """Parity with a tolerance, because the two operators are not bit-identical.

    SciPy provides a fused ``gaussian_laplace``; ``cupyx`` has none, so the GPU
    path composes ``laplace(gaussian_filter(...))``. Same operator, different
    floating-point ordering.
    """
    cpu, gpu = NumpyBackend(), CupyBackend()
    expected = cpu.to_numpy(cpu.gaussian_laplace(cpu.to_device(structured_tile), sigma))
    actual = gpu.to_numpy(gpu.gaussian_laplace(gpu.to_device(structured_tile), sigma))
    assert np.allclose(expected, actual, rtol=1e-4, atol=1e-4)


@requires_gpu
@pytest.mark.gpu
def test_gpu_synchronize_is_callable() -> None:
    CupyBackend().synchronize()
