"""The multi-GPU tiled consumer path.

These tests need at least two CUDA devices, and that is the point: with one GPU
every device id in the pool is 0, so a fold that ignores which card its arrays
live on lands on the right one by accident. Nothing here is reachable on a
single-GPU runner, which is why the defect below survived review and a green
suite.

Tiles are sharded round-robin over ``gpu_ids``, and a consumer that sets
``wants_device_mean`` / ``wants_device_focus`` is handed arrays still resident
on the tile's card. CuPy dispatches to whichever device is *current in the
calling thread*, not to the one an array lives on.

Three of the four consumers defend themselves: ``CentrePixelSampler`` and
``LabeledSumAccumulator`` open ``with array.device:`` before gathering, and
``BlockMeanAccumulator`` never touches CuPy. ``RoiOtsuSnrAccumulator`` does
not -- it infers ``on_gpu = isinstance(array, cp.ndarray)`` and then calls
``cp.asnumpy`` and ``roi_snr_db_batch(..., xp=cp)`` with no device context.

**What this is and is not.** Measured on two L4s before the fix,
``_run_tile`` handed that consumer a foreign-device array on 3 of 4 tiles. It
did not crash, because the operations it performs on the foreign array are
``cp.asnumpy``, which CuPy resolves cross-device. So this is a latent
mismatch, not an observed abort: the fold is one elementwise kernel or fancy
index away from ``cudaErrorIllegalAddress``, which aborts the interpreter
rather than raising.

The test therefore asserts the *invariant* rather than waiting for a crash --
during the fold, the current device must equal the device the folded arrays
live on. That is what fails before the fix and holds after, and it is what
keeps the next edit to that consumer from becoming a silent task death.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from spatialqc.backend import available_gpu_ids

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(len(available_gpu_ids()) < 2, reason="needs at least two CUDA devices"),
]

WINDOW = 35
TILE_MEM_BYTES = 64 * 1024 * 1024  # small on purpose, to force many tiles
SIDE = 1024
ROI = 128
# The map keys the tile pass actually produces; see the production consumer
# factory in image/qc.py. "focus_score" is a DataFrame column, not a map key.
SAMPLE_KEYS = ["focus_map", "mean_map"]


def _channel(side=SIDE, seed=0):
    """Nuclei-like DAPI content, the structure the focus metrics respond to."""
    rng = np.random.default_rng(seed)
    img = np.zeros((side, side), dtype=np.float32)
    yy, xx = np.mgrid[0:side, 0:side]
    for _ in range(400):
        cy, cx = rng.uniform(0, side), rng.uniform(0, side)
        rad = rng.uniform(3.0, 7.0)
        img += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad**2)))
    img += rng.normal(0, 0.02, (side, side)).astype(np.float32)
    return (np.clip(img, 0, None) / img.max() * 6000).astype(np.uint16)


def _roi_grid(side=SIDE, roi=ROI):
    starts = np.arange(0, side, roi, dtype=np.int64)
    yy, xx = np.meshgrid(starts, starts, indexing="ij")
    y1, x1 = yy.ravel(), xx.ravel()
    return y1, y1 + roi, x1, x1 + roi, y1 + roi // 2, x1 + roi // 2


def _run(channel, gpu_ids, spy_on_devices=False):
    """Run the tiled consumer path.

    With *spy_on_devices*, wrap ``RoiOtsuSnrAccumulator.consume`` to record the
    array's device against the thread's current device at fold time, then
    restore it. Returns ``(sampler, observations)``.
    """
    from spatialqc.image import qc

    shape = channel.shape
    y1, y2, x1, x2, cy, cx = _roi_grid(shape[0])
    sampler = qc.CentrePixelSampler(cy, cx, list(SAMPLE_KEYS))
    snr = qc.RoiOtsuSnrAccumulator(y1, y2, x1, x2, shape, map_key="mean_map")

    observations: list[tuple[int, int]] = []
    original = qc.RoiOtsuSnrAccumulator.consume

    if spy_on_devices:
        import cupy as cp

        lock = threading.Lock()

        def spy(self, tile_spec, maps):
            array = qc._device_or_host(maps, self.map_key)
            if qc._is_device_array(array):
                with lock:
                    observations.append((int(array.device.id), int(cp.cuda.Device().id)))
            return original(self, tile_spec, maps)

        qc.RoiOtsuSnrAccumulator.consume = spy

    try:
        qc._compute_channel_maps_tiled(
            channel,
            shape,
            WINDOW,
            list(gpu_ids),
            include_laplacian=False,
            gpu_mem_bytes=TILE_MEM_BYTES,
            consumers=[sampler, snr],
        )
    finally:
        qc.RoiOtsuSnrAccumulator.consume = original

    return sampler, observations


def test_the_fold_runs_on_the_card_its_arrays_live_on():
    """The invariant `_run_tile`'s own `finally` comment already claims.

    That comment says "the fold runs on this card". Before the device context
    was added it did not: the arrays came from the pooled card while device 0
    stayed current. Measured 3 mismatches out of 4 tiles on two L4s.
    """
    _, observations = _run(_channel(), available_gpu_ids()[:2], spy_on_devices=True)

    assert observations, "no device arrays reached the consumer; the test proves nothing"
    mismatched = [(arr, cur) for arr, cur in observations if arr != cur]
    assert not mismatched, (
        f"{len(mismatched)}/{len(observations)} folds ran on the wrong card "
        f"(array device, current device): {mismatched}"
    )
    # The sharding must actually have used more than one card, or the invariant
    # above is trivially satisfied and this test is vacuous.
    assert len({arr for arr, _ in observations}) > 1, (
        "every tile landed on one card; raise the tile count so the pool spreads"
    )


def test_every_roi_is_sampled_exactly_once_across_devices():
    """Sharding must not drop or double-count an ROI."""
    sampler, _ = _run(_channel(), available_gpu_ids()[:2])

    for key in SAMPLE_KEYS:
        values = sampler.values[key]
        assert values.size > 0, key
        unsampled = int((~np.isfinite(values)).sum())
        assert unsampled == 0, f"{key}: {unsampled} ROIs never sampled"


def test_sharding_across_devices_is_bit_identical_to_one_device():
    """Which card computed a tile must not change the number it produces.

    Image QC shards tiles over every visible GPU, so two ROIs of one sample are
    routinely graded on different cards. If results depended on the card, a
    sample's focus statistics would depend on how many GPUs the Batch instance
    happened to have.
    """
    channel = _channel(seed=1)
    ids = available_gpu_ids()[:2]

    one_sampler, _ = _run(channel, [ids[0]])
    many_sampler, _ = _run(channel, ids)

    for key in SAMPLE_KEYS:
        np.testing.assert_array_equal(
            one_sampler.values[key],
            many_sampler.values[key],
            err_msg=f"{key} differs between 1 GPU and {len(ids)} GPUs",
        )
