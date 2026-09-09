"""Close plan section 1 on REAL Xenium DAPI, not synthetic blobs.

Section 1 asks whether replacing the GPU-only LoG shim with the fused operator
moves the sample-level blurry-tile percentage that `focus_warn: 0.40` and
`focus_fail: 0.60` gate on. Synthetic mosaics established the mechanism (the
log-space shift is non-uniform, so the 2D GMM is not invariant) but cannot say
where a real sample sits relative to an empirically calibrated cutoff.

This runs the real production functions on real morphology images pulled from
the bundles used by actual Tower runs, mirroring the production path:
  * lap_var computed over a whole window, then sampled at tile CENTRES
    (image/qc.py:5170)
  * focus_score = std**2 / mean of raw intensity (image/qc.py:4608)
  * the 2D GMM via fit_focus_gmm_2d + classify_roi_blur_2d

The only difference between the two feature tables is which operator produced
dapi_lap_var.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

SRC = (
    "/home/projects/nextflow-tower-pipeline-dev/nf-core/spatialxe"
    "/.claude/worktrees/spatialqc-package/packages/spatialqc/src"
)
sys.path.insert(0, SRC)

import logging  # noqa: E402

logging.disable(logging.INFO)

import cupy as cp  # noqa: E402
from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter  # noqa: E402
from cupyx.scipy.ndimage import laplace as cp_laplace  # noqa: E402
from cupyx.scipy.ndimage import uniform_filter as cp_uniform_filter  # noqa: E402

from spatialqc.image.qc import (  # noqa: E402
    classify_roi_blur_2d,
    compute_laplacian_variance_map,
    fit_focus_gmm_2d,
)

REAL = Path(
    "/tmp/claude-470214627/-home-projects-nextflow-tower-pipeline-dev-nf-core-spatialxe"
    "/b9aed468-da0b-44bc-b7da-e6019628d1a5/scratchpad/real"
)
SAMPLES = {
    "v1_R2_control": REAL / "v1_R2_control_dapi.ome.tif",
    "atera_breast": REAL / "atera_breast_dapi.ome.tif",
}

TILE = 128  # tile side, px
WINDOW = 35  # focus window, production default
LAP_SIGMA = 1.0  # production default
GRID = 16  # GRID x GRID tiles per window -> 2048 px windows
N_WINDOWS = 6  # windows sampled across the section
BLUR_COL = "is_blurred_gmm_2d"
MIN_TISSUE_MEAN = 200  # a window whose mean is below this is empty slide


def dapi_level0(path):
    """Full-resolution DAPI as a 2-D array.

    tifffile's ``aszarr`` bridge requires zarr>=3, and spatialqc pins zarr<3
    deliberately, so lazy windowed reads are unavailable in this environment.
    Each downloaded morphology_focus file holds exactly one page -- the
    full-resolution channel -- so read that page directly and slice in memory.
    Reading the OME *series* instead would zero-fill the three sibling channel
    files that were not downloaded and cost 4x the memory.
    """
    with tifffile.TiffFile(path) as tf:
        return tf.pages[0].asarray()


def pick_windows(arr, rng, side):
    """Windows biased onto tissue: sample candidates, keep the brightest."""
    h, w = arr.shape
    candidates = []
    for _ in range(N_WINDOWS * 8):
        y = int(rng.integers(0, h - side))
        x = int(rng.integers(0, w - side))
        # Cheap probe on a strided read before committing to the full window.
        probe = arr[y : y + side : 16, x : x + side : 16]
        candidates.append((float(probe.mean()), y, x))
    candidates.sort(reverse=True)
    return [(y, x) for m, y, x in candidates[:N_WINDOWS] if m >= MIN_TISSUE_MEAN]


def shim_lap_var_map(image, gpu_id=0):
    """The operator that was REMOVED: laplace(gaussian_filter(x)) on GPU."""
    with cp.cuda.Device(gpu_id):
        image_f = cp.asarray(image.astype(np.float32, copy=False))
        lap = cp_laplace(cp_gaussian_filter(image_f, sigma=LAP_SIGMA)).astype(cp.float64)
        lap_mean = cp_uniform_filter(lap, size=WINDOW)
        lap_sq_mean = cp_uniform_filter(lap * lap, size=WINDOW)
        return cp.asnumpy(cp.maximum(lap_sq_mean - lap_mean**2, 0.0).astype(cp.float32))


def tiles_from_window(window):
    """Per-tile focus_score, mean intensity, and tile-centre coordinates."""
    focus, inten, cys, cxs = [], [], [], []
    for r in range(GRID):
        for c in range(GRID):
            blk = window[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].astype(np.float64)
            m, s = blk.mean(), blk.std()
            focus.append((s * s) / m if m > 0 else 0.0)
            inten.append(m)
            cys.append(r * TILE + TILE // 2)
            cxs.append(c * TILE + TILE // 2)
    return (
        np.array(focus),
        np.array(inten),
        np.array(cys),
        np.array(cxs),
    )


def blurry_flags(focus, inten, lap_var):
    df = pd.DataFrame(
        {
            "dapi_focus_score": focus,
            "dapi_lap_var": lap_var,
            "dapi_intensity": inten,
            "tissue_coverage": np.ones_like(focus),
        }
    )
    gmm, blur_idx = fit_focus_gmm_2d(df, focus_col_name="dapi_focus_score")
    out = classify_roi_blur_2d(df, gmm, blur_idx, focus_col_name="dapi_focus_score")
    return out[BLUR_COL].to_numpy().astype(bool)


def verdict(frac):
    return "FAIL" if frac >= 0.60 else ("WARN" if frac >= 0.40 else "PASS")


def main():
    side = GRID * TILE
    print(
        f"cupy {cp.__version__} | tile={TILE}px window={side}px "
        f"lap_sigma={LAP_SIGMA} windows/sample={N_WINDOWS}"
    )
    print("YAML cutoffs: focus_warn=0.40 focus_fail=0.60\n")

    for name, path in SAMPLES.items():
        if not path.exists():
            print(f"--- {name}: MISSING {path}")
            continue
        arr = dapi_level0(path)
        rng = np.random.default_rng(0)
        windows = pick_windows(arr, rng, side)
        print(f"=== {name}  shape={arr.shape}  windows on tissue={len(windows)}")

        all_focus, all_inten, all_fused, all_shim = [], [], [], []
        for y, x in windows:
            window = arr[y : y + side, x : x + side]
            focus, inten, cys, cxs = tiles_from_window(window)
            fused_map = compute_laplacian_variance_map(
                window, WINDOW, use_gpu=True, gpu_id=0, lap_sigma=LAP_SIGMA
            )
            shim_map = shim_lap_var_map(window)
            all_focus.append(focus)
            all_inten.append(inten)
            all_fused.append(fused_map[cys, cxs].astype(np.float64))
            all_shim.append(shim_map[cys, cxs].astype(np.float64))
            del window, fused_map, shim_map

        focus = np.concatenate(all_focus)
        inten = np.concatenate(all_inten)
        lap_fused = np.concatenate(all_fused)
        lap_shim = np.concatenate(all_shim)

        ratio = lap_fused / lap_shim
        shift = np.log1p(lap_fused) - np.log1p(lap_shim)
        print(f"  tiles={len(focus)}  intensity median={np.median(inten):.0f}")
        print(
            f"  fused/shim ratio  median {np.median(ratio):.4f}  "
            f"IQR [{np.percentile(ratio, 25):.4f}, {np.percentile(ratio, 75):.4f}]  "
            f"max {np.nanmax(ratio):.4f}"
        )
        print(f"  log shift         mean {shift.mean():+.4f}  sd {shift.std():.4f}")

        f_fused = blurry_flags(focus, inten, lap_fused)
        f_shim = blurry_flags(focus, inten, lap_shim)
        vf, vs = f_fused.mean(), f_shim.mean()
        print(
            f"  blurry fraction   fused {vf:.4f} ({verdict(vf)})   "
            f"shim {vs:.4f} ({verdict(vs)})   delta {100 * (vf - vs):+.2f} pp"
        )
        print(f"  relabelled        {int((f_fused != f_shim).sum())}/{len(f_fused)} tiles")
        crosses = verdict(vf) != verdict(vs)
        print(f"  VERDICT CHANGES?  {'YES -- recalibrate' if crosses else 'no'}\n")


if __name__ == "__main__":
    main()
