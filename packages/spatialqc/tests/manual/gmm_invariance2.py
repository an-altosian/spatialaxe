"""Does the fused-vs-shim LoG change move the sample-level blurry-tile %?

Plan doc section 1 blocking decision. `focus_warn: 0.40` / `focus_fail: 0.60`
are ABSOLUTE cutoffs on the blurry-tile fraction from the 2D GMM, calibrated
empirically -- most likely against the shim, since image QC runs under
`label 'process_gpu_qc'`.

Runs the REAL production functions (`fit_focus_gmm_2d` + `classify_roi_blur_2d`)
on two feature tables identical except that `dapi_lap_var` comes from the fused
operator in one and the removed shim in the other. Mirrors production exactly:
lap_var map computed over the whole mosaic then sampled at tile CENTRES
(qc.py:5170); focus_score = std**2/mean of raw intensity (qc.py:4608).

Reports, per seed:
  * the fused/shim ratio distribution and its log-space shift (mean and sd --
    the sd is the cloud DEFORMATION, which is what moves tiles across the GMM
    boundary; a uniform ratio would be a pure translation and thus harmless)
  * spearman(blur_sigma, log_shift) -- the SIGN says whether fused widens or
    compresses blur/focus separation
  * pearson(focus_score, lap_var) both ways -- this is what
    `lap_focus_corr_warn/fail` gate on
  * the blurry-tile fraction both ways, against the YAML cutoffs
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Resolve the package source relative to this file, so the script keeps working
# after the package is extracted to its own repository.
SRC = str(Path(__file__).resolve().parents[2] / "src")
sys.path.insert(0, SRC)

import logging  # noqa: E402

logging.disable(logging.INFO)

import cupy as cp  # noqa: E402
from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter  # noqa: E402
from cupyx.scipy.ndimage import laplace as cp_laplace  # noqa: E402
from cupyx.scipy.ndimage import uniform_filter as cp_uniform_filter  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from spatialqc.image.qc import (  # noqa: E402
    classify_roi_blur_2d,
    compute_laplacian_variance_map,
    fit_focus_gmm_2d,
)

TILE = 128
GRID = 12
WINDOW = 35
LAP_SIGMA = 1.0
BLUR_COL = "is_blurred_gmm_2d"


def build_mosaic(rng):
    blur_levels = np.concatenate(
        [
            np.zeros(GRID * GRID - 44),
            rng.uniform(0.3, 0.9, 16),
            rng.uniform(1.0, 2.0, 16),
            rng.uniform(2.5, 4.0, 12),
        ]
    )
    rng.shuffle(blur_levels)

    mosaic = np.zeros((GRID * TILE, GRID * TILE), dtype=np.float32)
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    for k, sigma in enumerate(blur_levels):
        r, c = divmod(k, GRID)
        tile = np.zeros((TILE, TILE), dtype=np.float32)
        for _ in range(int(rng.integers(18, 34))):
            cy, cx = rng.uniform(6, TILE - 6, 2)
            rad = rng.uniform(3.0, 6.0)
            tile += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * rad**2)))
        tile += rng.normal(0, 0.03, (TILE, TILE)).astype(np.float32)
        if sigma > 0:
            tile = gaussian_filter(tile, sigma=sigma)
        mosaic[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE] = tile

    mosaic = np.clip(mosaic, 0, None)
    return (mosaic / mosaic.max() * 8000.0).astype(np.uint16), blur_levels


def shim_lap_var_map(image, gpu_id=0):
    """The REMOVED operator: laplace(gaussian_filter(x)) on GPU."""
    with cp.cuda.Device(gpu_id):
        image_f = cp.asarray(image.astype(np.float32, copy=False))
        lap = cp_laplace(cp_gaussian_filter(image_f, sigma=LAP_SIGMA)).astype(cp.float64)
        lap_mean = cp_uniform_filter(lap, size=WINDOW)
        lap_sq_mean = cp_uniform_filter(lap * lap, size=WINDOW)
        return cp.asnumpy(cp.maximum(lap_sq_mean - lap_mean**2, 0.0).astype(cp.float32))


def tile_centres():
    idx = np.arange(GRID) * TILE + TILE // 2
    cy, cx = np.meshgrid(idx, idx, indexing="ij")
    return cy.ravel(), cx.ravel()


def per_tile_focus(mosaic):
    focus, inten = [], []
    for k in range(GRID * GRID):
        r, c = divmod(k, GRID)
        blk = mosaic[r * TILE : (r + 1) * TILE, c * TILE : (c + 1) * TILE].astype(np.float64)
        m, s = blk.mean(), blk.std()
        focus.append((s * s) / m if m > 0 else 0.0)
        inten.append(m)
    return np.array(focus), np.array(inten)


def blurry_fraction(focus, inten, lap_var):
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


def spearman(a, b):
    return pd.Series(a).corr(pd.Series(b), method="spearman")


def pearson(a, b):
    return pd.Series(a).corr(pd.Series(b), method="pearson")


def verdict(frac):
    return "FAIL" if frac >= 0.60 else ("WARN" if frac >= 0.40 else "PASS")


def main():
    print(f"cupy {cp.__version__}  tiles={GRID * GRID}  tile={TILE}px  lap_sigma={LAP_SIGMA}")
    print(
        "YAML cutoffs: focus_warn=0.40 focus_fail=0.60 | "
        "lap_focus_corr_warn=0.50 lap_focus_corr_fail=0.25\n"
    )

    cy, cx = tile_centres()
    rows = []

    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        mosaic, blur_levels = build_mosaic(rng)
        focus, inten = per_tile_focus(mosaic)

        lap_fused = compute_laplacian_variance_map(
            mosaic, WINDOW, use_gpu=True, gpu_id=0, lap_sigma=LAP_SIGMA
        )[cy, cx].astype(np.float64)
        lap_shim = shim_lap_var_map(mosaic)[cy, cx].astype(np.float64)

        ratio = lap_fused / np.maximum(lap_shim, 1e-12)
        shift = np.log1p(lap_fused) - np.log1p(lap_shim)

        f_fused = blurry_fraction(focus, inten, lap_fused)
        f_shim = blurry_fraction(focus, inten, lap_shim)

        # Sign test: is the operator difference systematic with defocus?
        sp_blur_shift = spearman(blur_levels, shift)
        corr_fused = pearson(focus, lap_fused)
        corr_shim = pearson(focus, lap_shim)

        print(f"--- seed {seed} " + "-" * 60)
        print(
            f"  ratio fused/shim   median {np.median(ratio):.4f}  "
            f"IQR [{np.percentile(ratio, 25):.4f}, {np.percentile(ratio, 75):.4f}]  "
            f"max {ratio.max():.4f}"
        )
        print(
            f"  log shift          mean {shift.mean():+.4f}  sd {shift.std():.4f}  "
            f"(sd = cloud deformation)"
        )
        print(
            f"  spearman(blur_sigma, log_shift) = {sp_blur_shift:+.4f}  "
            f"-> fused {'WIDENS' if sp_blur_shift < 0 else 'COMPRESSES'} blur/focus separation"
        )
        print(
            f"  lap_focus_corr     fused {corr_fused:+.4f}  shim {corr_shim:+.4f}  "
            f"delta {corr_fused - corr_shim:+.4f}"
        )
        print(
            f"  blurry fraction    fused {f_fused.mean():.4f} ({verdict(f_fused.mean())})  "
            f"shim {f_shim.mean():.4f} ({verdict(f_shim.mean())})  "
            f"delta {100 * (f_fused.mean() - f_shim.mean()):+.2f} pp"
        )
        print(f"  label disagreement {int((f_fused != f_shim).sum())}/{len(f_fused)} tiles")

        rows.append(
            {
                "seed": seed,
                "delta_pp": 100 * (f_fused.mean() - f_shim.mean()),
                "fused": f_fused.mean(),
                "shim": f_shim.mean(),
                "sp": sp_blur_shift,
                "dcorr": corr_fused - corr_shim,
            }
        )

    print("\n" + "=" * 74)
    d = np.array([r["delta_pp"] for r in rows])
    print(f"blurry-fraction delta across 3 seeds: {np.round(d, 2).tolist()} pp")
    print(
        f"  mean {d.mean():+.2f} pp   consistent sign: {bool(np.all(np.sign(d) == np.sign(d[0])))}"
    )
    sp = np.array([r["sp"] for r in rows])
    print(
        f"spearman(blur, shift): {np.round(sp, 4).tolist()}  "
        f"consistent sign: {bool(np.all(np.sign(sp) == np.sign(sp[0])))}"
    )
    dc = np.array([r["dcorr"] for r in rows])
    print(f"lap_focus_corr delta:  {np.round(dc, 4).tolist()}")


if __name__ == "__main__":
    main()
