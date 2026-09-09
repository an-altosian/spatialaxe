# spatialqc: first GPU validation of the ported QC package

Status: the GPU code paths in `spatialqc` have now been executed for the first time.

## Why this document exists

The session that created `spatialqc` had **no GPU access**.
The machine's `/dev/nvidia*` nodes were created at 17:23; that session ended at 17:21.
Every CuPy branch in the package was therefore written, reviewed and merged without ever running, and the "236 tests passing" figure was a CPU-only number.

This document records what happened when those paths were finally executed on hardware.

## Environment

| Item     | Value                                                     |
| -------- | --------------------------------------------------------- |
| Hardware | 4x NVIDIA L4, 22 GiB each (~14 GiB free; GPUs are shared) |
| Driver   | 580.159.03                                                |
| CuPy     | 14.0.1                                                    |
| Python   | 3.11, env `xenium-test-local`                             |
| Device   | `CUDA_VISIBLE_DEVICES=0` for all runs                     |

Note: the sandbox masks `/dev` and `/proc`, so every GPU command must run with the sandbox disabled.
A sandboxed shell reports `/dev/nvidia*` as *"No such file or directory"* rather than as a permission error, so absence of evidence inside the sandbox is not evidence of absence.

## 1. Test suite with CuPy present

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src \
  python -m pytest tests/ -n 8 -q
```

Result: **245 passed, 1 skipped**, exit 0.

Of these, **9 GPU-marked tests executed for the first time and all passed** (`pytest -m gpu` reports `9 passed, 236 deselected`).
The single skip is `tests/test_backend.py:56`, whose skip reason is *"a CUDA device is present"* — it covers the no-CuPy error path and is correctly inactive here.
The suite therefore covers both directions; this machine exercises the GPU direction, CI exercises the other.

## 2. The fused Laplacian-of-Gaussian change (plan section 1)

Commit `62615ac` replaced the GPU-only shim `laplace(gaussian_filter(x, sigma))` with the fused `cupyx.scipy.ndimage.gaussian_laplace(x, sigma)`, matching the CPU path.

### 2.1 `cupyx.scipy.ndimage.gaussian_laplace` exists

Verified directly on CuPy 14.0.1, not from documentation.
The removed shim's justifying comment — that CuPy exposes no fused `gaussian_laplace` — was false.

### 2.2 CPU and GPU now agree

`compute_laplacian_variance_map` called on the same tile through both branches:

| Metric              | Value        |
| ------------------- | ------------ |
| Correlation         | 1.00000000   |
| Max relative devi.  | 0.0000%      |
| Mean ratio GPU/CPU  | 1.00000012   |

This closes the defect the plan flagged as most serious: the GPU-OOM fallback switches to the CPU path mid-run, so before this fix a single sample could be graded on two different scales.
It cannot any more.

### 2.3 The shim's divergence is content-dependent

The magnitude of the fused-vs-shim difference depends on image content, which is why three measurements disagree.
All are at the production `lap_sigma` of 1.0.

| Tile content                            | Max rel. dev. | Correlation | var(shim)/var(fused) |
| --------------------------------------- | ------------- | ----------- | -------------------- |
| Hard edges + sinusoid texture + noise   | 25.2%         | 0.9997      | 0.612                |
| Nuclei-like Gaussian blobs (DAPI-like)  | ~7% median    | —           | —                    |
| Plan doc's original synthetic tile      | 16.6%         | 0.9937      | 0.77                 |

These are three different tiles, not a correction of one another.
The operative figure for DAPI morphology images is the nuclei-like row: a median `fused/shim` ratio of **1.067–1.071**, IQR roughly [1.043, 1.098], with a tail to ~1.44.

### 2.4 The shift is systematically larger on sharp tiles

This is the most useful new result.
Across three independent 144-tile mosaics:

| Seed | `spearman(blur_sigma, log_shift)` |
| ---- | --------------------------------- |
| 0    | -0.756                            |
| 1    | -0.740                            |
| 2    | -0.725                            |

The sign is consistent and the magnitude is stable.
A negative correlation means the fused operator raises `lap_var` **more on sharp tiles than on blurry ones**, so it *widens* the blur/focus separation rather than compressing it.

That is the safe direction, and it is an argument for the fix on its own merits: the fused operator is not merely more consistent, it is more discriminative.

### 2.5 `lap_focus_corr` thresholds are unaffected

`lap_focus_corr_warn: 0.50` and `lap_focus_corr_fail: 0.25` gate the correlation between `focus_score` and `lap_var`.

| Seed | corr (fused) | corr (shim) | Delta   |
| ---- | ------------ | ----------- | ------- |
| 0    | +0.3932      | +0.3832     | +0.0100 |
| 1    | +0.2320      | +0.2287     | +0.0034 |
| 2    | +0.1995      | +0.1926     | +0.0069 |

The deltas are within +0.01 and all slightly positive.
These two thresholds need no attention.

### 2.6 `focus_median_warn` is not downstream of the change

`focus_median_warn: 100` is an absolute cutoff, but it applies to `focus_score`, which is `std**2 / mean` of the **raw intensity** (`image/qc.py:4608`).
It never touches the Laplacian.
Unaffected by construction.

### 2.7 The blurry-tile fraction is a variance risk, not a bias

This is the part that still needs real samples, and the reason is not what the plan assumed.

The 2D GMM (`fit_focus_gmm_2d`) fits `[log1p(focus_score), log1p(lap_var)]` with `covariance_type="full"` and **no feature scaling**.
A *uniform* rescaling of `lap_var` would become a pure translation in log space, and a full-covariance GMM is translation-equivariant, so it would be harmless.
The rescaling is not uniform: the log-space shift has mean ~0.078 but standard deviation ~0.058.
That standard deviation is a deformation of the 2D cloud, and it can move tiles across the component boundary.

Running the real `fit_focus_gmm_2d` + `classify_roi_blur_2d` on feature tables identical except for the operator:

| Seed | Blurry (fused) | Blurry (shim) | Delta      | Tiles relabelled |
| ---- | -------------- | ------------- | ---------- | ---------------- |
| 0    | 0.1806         | 0.3264        | -14.58 pp  | 21 / 144         |
| 1    | 0.2708         | 0.2500        | +2.08 pp   | 5 / 144          |
| 2    | 0.1458         | 0.1528        | -0.69 pp   | 1 / 144          |

The sign is **not** consistent and the mean is -4.40 pp.
So there is no systematic recalibration factor to apply — the cutoffs are not uniformly mis-scaled.

What the table does show is that on a sample whose focus/blur bimodality is marginal (seed 0), a small deformation can swing the fraction by ~15 pp.
`focus_warn` is 0.40, and the threshold YAML records an observed blurry floor of ~27% even on good samples, so real samples sit close enough to the warn boundary that a 15 pp swing can cross it.

The risk is therefore **GMM instability on marginal samples**, not a scale shift in the cutoffs.

## 3. What still needs real samples

Everything above uses synthetic mosaics: nuclei-like Gaussian blobs, `lap_var` sampled at tile centres as production does (`image/qc.py:5170`), `focus_score` computed per tile as `std**2/mean`.
Synthetic data establishes direction and mechanism; it cannot establish where a real sample sits relative to an empirically calibrated cutoff.

To close plan section 1, run image QC on 2-3 calibration samples and compare, fused vs shim:

- `pct_blurred_gmm_2d_roi` (the blurry-tile fraction) against `focus_warn: 0.40` / `focus_fail: 0.60`
- `pct_blurred_gmm_2d_roi_warn: 20.0` (thresholds YAML line 338), which is also downstream of the GMM fraction
- `lap_focus_corr` against its two cutoffs, as a cheap confirmation of section 2.5 on real data

A sample that changes verdict is the signal to recalibrate; a sample that does not is evidence the change is safe to ship as-is.

The scripts used here are reproducible and take about a minute each on one L4:

```bash
# CPU-vs-GPU agreement, and fused-vs-shim divergence across lap_sigma
python log_gpu_check.py
# 3-seed GMM stability, blur-correlation sign, lap_focus_corr deltas
python gmm_invariance2.py
```

## 4. Environment note for anyone reproducing this

`xenium-test-local` is missing `esda`, `libpysal` and `nsitk`.
The unit tests stub all three, so the suite passes without them, but a full end-to-end image QC run needs `nsitk` (`generate_tissue_mask` uses it on the path every run takes) and `esda`/`libpysal` (Moran's I for the negative-probe SNR metric).
It also carries `pyarrow 21.0.0`, which is above the cap anndata has previously required; that matters for transcript QC but not for the image QC paths exercised here.
