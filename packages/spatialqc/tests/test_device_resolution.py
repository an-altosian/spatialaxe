"""Tests for the CPU/GPU mode selection inside the image QC analysis.

:mod:`spatialqc.backend` is covered separately in ``test_backend.py``. What is
covered here is the analysis-level selector: ``resolve_available_gpus`` is the
single point where ``run_image_qc`` decides which devices the run may use, so it
is what actually enforces ``--device``.

It exists as a module-level function precisely so this behaviour is testable;
inline in ``run_image_qc`` it would have needed a real Xenium bundle to reach.
"""

from __future__ import annotations

import pytest

from spatialqc.image.qc import resolve_available_gpus


def test_cpu_returns_no_devices_even_when_gpus_are_visible(monkeypatch) -> None:
    """--device cpu must win over autodetection.

    This is the case the previous interface could not express: --max-gpus 0 did
    not force the CPU path, because the truncation is guarded by `and max_gpus`
    and 0 is falsy, so the full device list survived.
    """
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [0, 1, 2, 3])
    assert resolve_available_gpus("cpu") == []


def test_gpu_raises_when_no_device_is_usable(monkeypatch) -> None:
    """No silent fallback: an explicit GPU request must fail loudly.

    A run that quietly drops to CPU returns a correct-looking result at roughly
    50x the cost, on a node that was requested precisely because it had a GPU.
    """
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [])
    with pytest.raises(RuntimeError, match="no usable CUDA device"):
        resolve_available_gpus("gpu")


def test_gpu_returns_the_devices_when_present(monkeypatch) -> None:
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [0, 1])
    assert resolve_available_gpus("gpu") == [0, 1]


def test_auto_falls_back_to_cpu_without_raising(monkeypatch) -> None:
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [])
    assert resolve_available_gpus("auto") == []


def test_auto_uses_gpus_when_present(monkeypatch) -> None:
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [0])
    assert resolve_available_gpus("auto") == [0]


@pytest.mark.parametrize("selector", ["", "tpu", "GPU", None])
def test_unknown_selector_raises_value_error(selector) -> None:
    with pytest.raises(ValueError, match="device must be one of"):
        resolve_available_gpus(selector)


# ---------------------------------------------------------------------------
# max_gpus capping
# ---------------------------------------------------------------------------


def test_max_gpus_caps_the_device_list(monkeypatch) -> None:
    """A task granted one GPU must not use all four on the instance.

    Nextflow's `accelerator` directive sizes the AWS Batch request only; it does
    not restrict CUDA visibility.
    """
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [0, 1, 2, 3])
    assert resolve_available_gpus("auto", max_gpus=1) == [0]
    assert resolve_available_gpus("auto", max_gpus=2) == [0, 1]


@pytest.mark.parametrize("no_cap", [0, None])
def test_zero_or_none_max_gpus_means_no_cap(monkeypatch, no_cap) -> None:
    # Documented as "0 = use every device detected"; kept for compatibility with
    # the existing --max-gpus contract.
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [0, 1, 2])
    assert resolve_available_gpus("auto", max_gpus=no_cap) == [0, 1, 2]


def test_cap_larger_than_the_device_count_is_harmless(monkeypatch) -> None:
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [0])
    assert resolve_available_gpus("auto", max_gpus=8) == [0]


def test_cap_is_applied_to_an_explicit_gpu_request(monkeypatch) -> None:
    monkeypatch.setattr("spatialqc.image.qc.detect_gpu_ids", lambda: [0, 1, 2, 3])
    assert resolve_available_gpus("gpu", max_gpus=2) == [0, 1]
