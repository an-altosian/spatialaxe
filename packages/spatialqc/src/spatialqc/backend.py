"""CPU and GPU compute backends for the image QC filter kernels.

The QC pipeline runs on two very different hosts: GPU nodes (AWS Batch
``g6e``-class instances, CuPy + CUDA 12) and CPU-only nodes. Both paths must be
first-class, so this module resolves the choice **once** into a :class:`Backend`
object instead of scattering ``if use_gpu:`` branches through the numerics.

Design rules (see the ``python-packaging`` skill, RULE P5):

* Each backend calls its **native** primitives. The GPU path uses
  ``cupyx.scipy.ndimage``; the CPU path uses ``scipy.ndimage`` (which is itself
  a compiled C implementation) plus Numba kernels where a fused reduction beats
  a filter chain. Neither emulates the other.
* ``cupy`` is imported in exactly one place -- here -- behind a broad guard.
* ``device="gpu"`` with no usable CUDA device **raises**. A silent fallback to
  CPU produces a correct-looking result at roughly 50x the cost, on a node that
  was paid for precisely because it had a GPU.
* Host/device transfers happen at the edges (:meth:`Backend.to_device` /
  :meth:`Backend.to_numpy`), once per tile, never per operation.

Numerical note: the CPU Laplacian-of-Gaussian calls
``scipy.ndimage.gaussian_laplace`` directly, while the GPU path composes
``laplace(gaussian_filter(...))`` because ``cupyx`` exposes no fused
``gaussian_laplace``. These are mathematically the same operator but not
bit-identical, which is why the cross-backend parity test asserts
``allclose`` with a tolerance rather than equality.
"""

from __future__ import annotations

import logging
from types import ModuleType
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Backend",
    "CupyBackend",
    "Device",
    "NumpyBackend",
    "available_gpu_ids",
    "get_backend",
    "gpu_is_available",
]

logger = logging.getLogger(__name__)

#: Accepted values for the ``--device`` selector.
#:
#: ``auto`` picks the GPU when one is usable and the CPU otherwise, and is meant
#: for interactive use. Automated callers should pass ``cpu`` or ``gpu``
#: explicitly so that a missing driver fails loudly instead of quietly costing a
#: day of compute.
Device = Literal["auto", "cpu", "gpu"]


# ---------------------------------------------------------------------------
# Optional CuPy import
# ---------------------------------------------------------------------------
# The guard is deliberately broad. ``cupy-cuda12x`` is installed in the QC
# container, so importing it on a host with no NVIDIA driver raises
# ``RuntimeError`` / ``CUDARuntimeError`` rather than ``ImportError``; catching
# only ``ImportError`` would abort the run on every CPU-only node.
try:  # pragma: no cover - depends on host hardware
    import cupy as _cp
    from cupyx.scipy.ndimage import gaussian_filter as _cupy_gaussian_filter
    from cupyx.scipy.ndimage import laplace as _cupy_laplace
    from cupyx.scipy.ndimage import uniform_filter as _cupy_uniform_filter

    HAS_CUPY = True
except Exception:  # noqa: BLE001 - see comment above
    _cp = None  # type: ignore[assignment]
    _cupy_gaussian_filter = None  # type: ignore[assignment]
    _cupy_laplace = None  # type: ignore[assignment]
    _cupy_uniform_filter = None  # type: ignore[assignment]
    HAS_CUPY = False


def gpu_is_available() -> bool:
    """Report whether a CUDA device can actually be used.

    ``HAS_CUPY`` only says the module imported. This additionally queries the
    CUDA runtime, so a container that has CuPy installed but no GPU attached is
    correctly reported as unavailable.
    """
    return bool(available_gpu_ids())


def available_gpu_ids(max_gpus: int | None = None) -> list[int]:
    """Return usable CUDA device IDs, optionally capped.

    Args:
        max_gpus: Upper bound on the number of devices returned. Needed because
            an AWS Batch ``accelerator`` request only tells the scheduler how
            many GPUs to reserve -- it does not restrict what CUDA can see, so a
            task asking for one GPU that lands on a four-GPU instance would
            otherwise detect and use all four.

    Returns:
        Device IDs, or an empty list when no GPU is usable.
    """
    if not HAS_CUPY:
        logger.info("CuPy is not importable; GPU backend unavailable.")
        return []
    try:
        n_devices = _cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001
        # CuPy imported but the runtime could not be queried (no driver, GPU not
        # attached to the container, init failure). Log it: otherwise this is
        # indistinguishable from "no GPU present", and a silent CPU fallback on a
        # GPU node looks like correct behaviour.
        logger.warning(
            "CuPy is installed but GPU detection failed (%s: %s); treating as CPU-only.",
            type(exc).__name__,
            exc,
        )
        return []
    ids = list(range(n_devices))
    if max_gpus is not None:
        ids = ids[:max_gpus]
    return ids


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Backend(Protocol):
    """The filter primitives the focus/SNR map kernels need from a device.

    Implementations are stateless apart from the CUDA device they target, so a
    backend can be created per worker process and reused across tiles.
    """

    #: Short label used in logs and in the metrics JSON (``"cpu"`` / ``"gpu"``).
    name: str
    #: The array module: :mod:`numpy` or :mod:`cupy`. Passed to array-module
    #: generic kernels written as ``f(x, xp)`` so one implementation serves both.
    xp: ModuleType

    def to_device(self, image: NDArray[Any]) -> Any:
        """Move a host array onto the device (a no-op on CPU)."""
        ...

    def to_numpy(self, array: Any) -> NDArray[np.float32]:
        """Bring an array back to the host as ``float32``."""
        ...

    def uniform_filter(self, array: Any, size: int) -> Any:
        """Box filter, used for the local mean and local mean-of-squares."""
        ...

    def laplace(self, array: Any) -> Any:
        """Discrete Laplacian, used for the sharpness response."""
        ...

    def gaussian_laplace(self, array: Any, sigma: float) -> Any:
        """Laplacian of Gaussian at the given sigma."""
        ...

    def synchronize(self) -> None:
        """Block until queued device work completes (a no-op on CPU).

        Required before timing a tile or reading a result, because CUDA calls are
        asynchronous and would otherwise report the enqueue time.
        """
        ...


class NumpyBackend:
    """CPU backend built on :mod:`scipy.ndimage`.

    ``scipy.ndimage`` is a compiled C implementation and is the fastest
    general-purpose option available on the host; it is not a fallback stub.
    Callers that need a fused reduction over many small tiles should prefer the
    Numba kernels in :mod:`spatialqc.image.snr`, which avoid materialising the
    intermediate filter outputs entirely.
    """

    name = "cpu"

    def __init__(self) -> None:
        # scipy lives in the `image` extra, not the core dependency set, so it is
        # imported lazily here rather than at module scope.
        from scipy import ndimage

        self.xp: ModuleType = np
        self._ndimage = ndimage

    def to_device(self, image: NDArray[Any]) -> NDArray[Any]:
        return image

    def to_numpy(self, array: Any) -> NDArray[np.float32]:
        return np.asarray(array, dtype=np.float32)

    def uniform_filter(self, array: Any, size: int) -> Any:
        return self._ndimage.uniform_filter(array, size=size)

    def laplace(self, array: Any) -> Any:
        return self._ndimage.laplace(array)

    def gaussian_laplace(self, array: Any, sigma: float) -> Any:
        # scipy provides the fused operator directly, which is both faster and
        # more accurate than smoothing and differentiating in two passes.
        return self._ndimage.gaussian_laplace(array, sigma=sigma)

    def synchronize(self) -> None:
        return None

    def __repr__(self) -> str:
        return "NumpyBackend()"


class CupyBackend:
    """GPU backend built on ``cupyx.scipy.ndimage``.

    Args:
        device_id: CUDA device to bind to. Multi-GPU runs create one backend per
            device and shard tiles across them.
    """

    name = "gpu"

    def __init__(self, device_id: int = 0) -> None:
        if not HAS_CUPY:  # pragma: no cover - guarded by get_backend
            raise RuntimeError(
                "CupyBackend requires CuPy. Install the 'gpu' extra "
                "(pip install 'spatialqc[gpu]') or select device='cpu'."
            )
        self.xp: ModuleType = _cp
        self.device_id = device_id
        self._device = _cp.cuda.Device(device_id)

    def to_device(self, image: NDArray[Any]) -> Any:
        with self._device:
            return _cp.asarray(image)

    def to_numpy(self, array: Any) -> NDArray[np.float32]:
        return _cp.asnumpy(array).astype(np.float32)

    def uniform_filter(self, array: Any, size: int) -> Any:
        return _cupy_uniform_filter(array, size=size)

    def laplace(self, array: Any) -> Any:
        return _cupy_laplace(array)

    def gaussian_laplace(self, array: Any, sigma: float) -> Any:
        # cupyx exposes no fused gaussian_laplace, so smooth then differentiate.
        # This is the same operator as the SciPy call, up to floating-point
        # ordering -- hence the tolerance in the parity test.
        return _cupy_laplace(_cupy_gaussian_filter(array, sigma=sigma))

    def synchronize(self) -> None:
        self._device.synchronize()

    def __repr__(self) -> str:
        return f"CupyBackend(device_id={self.device_id})"


def get_backend(device: Device = "auto", *, device_id: int = 0) -> Backend:
    """Resolve a device selector into a concrete backend.

    Args:
        device: ``"cpu"`` forces the CPU path, ``"gpu"`` requires a usable CUDA
            device, and ``"auto"`` prefers the GPU when one is present.
        device_id: CUDA device to bind when the GPU path is selected.

    Returns:
        A :class:`Backend` implementation.

    Raises:
        RuntimeError: If ``device="gpu"`` but no CUDA device is usable. This is
            intentional: falling back silently would hide a misconfigured GPU
            node behind a run that merely looks slow.
        ValueError: If *device* is not one of the accepted selectors.
    """
    if device == "cpu":
        return NumpyBackend()
    if device == "gpu":
        ids = available_gpu_ids()
        if not ids:
            raise RuntimeError(
                "device='gpu' was requested but no usable CUDA device was found. "
                "Install the 'gpu' extra and attach a GPU, or pass device='cpu'."
            )
        return CupyBackend(device_id=device_id)
    if device == "auto":
        if available_gpu_ids():
            logger.info("device='auto' resolved to the GPU backend.")
            return CupyBackend(device_id=device_id)
        logger.info("device='auto' resolved to the CPU backend (no CUDA device).")
        return NumpyBackend()
    raise ValueError(f"device must be one of 'auto', 'cpu', 'gpu'; got {device!r}")


def sanitize(array: NDArray[np.float32]) -> NDArray[np.float32]:
    """Replace non-finite values with zero, in place.

    Filter chains on tile edges and on all-zero background produce NaN/Inf; the
    downstream reductions treat zero as "no signal", so this normalises them.
    """
    array[~np.isfinite(array)] = 0.0
    return array
