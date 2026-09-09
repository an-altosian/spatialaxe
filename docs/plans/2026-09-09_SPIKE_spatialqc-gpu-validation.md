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
| Device   | `0` for single-device runs; `0,1,2,3` for the multi-GPU tests |

Note: the sandbox masks `/dev` and `/proc`, so every GPU command must run with the sandbox disabled.
A sandboxed shell reports `/dev/nvidia*` as *"No such file or directory"* rather than as a permission error, so absence of evidence inside the sandbox is not evidence of absence.

## 1. Test suite with CuPy present

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src \
  python -m pytest tests/ -n 8 -q
```

Result on the tree as reviewed: **245 passed, 1 skipped**, exit 0.
After the two defect fixes below and their three regression tests: **247 passed, 1 skipped**, with 11 GPU-marked tests.

Of these, **9 GPU-marked tests executed for the first time and all passed** (`pytest -m gpu` reported `9 passed, 236 deselected`).

Both that run and the review agent's independent `244 passed` were taken with `CUDA_VISIBLE_DEVICES=0`, on the tree *before* the fixes in section 3.
Neither defect there is reachable on a single device, which is why the two figures agreed and why both looked clean.
The number that describes the current tree is 247, run with `CUDA_VISIBLE_DEVICES=0,1,2,3`.
The single skip is `tests/test_backend.py:61`, whose skip reason is *"a CUDA device is present"* — it covers the no-CuPy error path and is correctly inactive here.
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

This was the part that needed real samples, and the reason is not what the plan assumed. Section 4 settles it on real data; the mechanism below is why the answer was not obvious in advance.

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

## 3. Two defects found in the never-executed GPU code

Both were reproduced on hardware, both are fixed in commit `3c97443`, and **neither is reachable on a single-GPU host** — which is why they survived a review and a green suite.

### 3.1 `CupyBackend` ran its filters on whichever device was current

Only `to_device` entered `with self._device`.
`uniform_filter`, `laplace`, `gaussian_laplace` and `to_numpy` did not.
CuPy dispatches to the *current* device, not to the one the array lives on, so the documented multi-GPU pattern gave:

```text
device 0: OK
device 1: ValueError: The device where the array resides (1) is
          different from the current device (0)
```

All operations now route through one `_on_device` helper rather than four separate wrappings.
No test had ever passed a non-zero `device_id`.

### 3.2 A caller-side invariant fell out of fixing it

The arrays these methods return live on `device_id`, which is not the current device.
Touching one with a raw CuPy call outside a device context is an **unrecoverable `cudaErrorIllegalAddress` that aborts the interpreter** — not a catchable exception. With a backend bound to device 1:

| Call                                  | Outcome                          |
| ------------------------------------- | -------------------------------- |
| `backend.to_numpy(result).mean()`     | returns normally                 |
| `float(result.mean())`                | process aborts, no traceback     |

This is now in the `CupyBackend` docstring, and it is the reason section 4.2 of the plan needed correcting: the remaining raw `cp.` sites in `image/qc.py` are **not** device-scoped, and tiles are sharded round-robin across devices (`gpu_ids[slot % len(gpu_ids)]`, `image/qc.py:3215`).
There is no `cp.cuda.Device` context anywhere between lines 2200 and 3100, yet the `consume()` callbacks in that range knowingly accept device arrays (`isinstance(array, cp.ndarray)`, line 2941).
On a multi-GPU instance that is a live abort, not a tidiness problem.
Image QC has in practice received one GPU, where every device id is 0 and the path is unreachable.

### 3.3 `max_gpus=0` meant opposite things

```text
backend.available_gpu_ids(max_gpus=0)   -> []
qc.resolve_available_gpus("auto", 0)    -> [0, 1]
```

`--max-gpus` defaults to `0` and is documented as "use every device", and requesting the CPU is the `device` selector's job — that single control point is precisely why `--max-gpus 0` must not mean "no GPUs".
`available_gpu_ids` was the outlier and is now aligned.
`test_backend.py` had enshrined the inverted convention, so it was corrected in the same commit: left as it was, swapping one function for the other during the protocol migration would have made every default run silently CPU-only, which looks like correct output at roughly 50x the cost.

## 4. Closed on real samples: no recalibration needed

The bundles used by real Tower runs are listed in `tower_launch/*.csv`.
Two were pulled and their DAPI morphology channels compared, fused versus shim, through the real `fit_focus_gmm_2d` + `classify_roi_blur_2d`:

| Sample          | Platform    | Tiles | Blurry (fused) | Blurry (shim) | Delta        | Relabelled | Verdict     |
| --------------- | ----------- | ----- | -------------- | ------------- | ------------ | ---------- | ----------- |
| `v1_R2_control` | Xenium v1   | 1536  | 0.1439         | 0.1582        | **-1.43 pp** | 24 / 1536  | PASS → PASS |
| `atera_breast`  | Xenium v2   | 1536  | 0.2617         | 0.2591        | **+0.26 pp** | 4 / 1536   | PASS → PASS |

**Neither sample changes verdict, and both sit far below `focus_warn: 0.40`.**

Two things make this more than a pair of data points.

First, the deltas are an order of magnitude smaller than the synthetic worst case (-14.58 pp at seed 0). Real tissue does not reproduce the marginal-bimodality pathology that a contrived mosaic can: 24 and 4 tiles out of 1536 move, against 21 out of 144 synthetically.

Second, the harness independently reproduces a number the threshold YAML already records. `atera_breast` measures 25.9-26.2% blurry, and the YAML's own comment says *"Observed tissue-filtered % blurry floor: ~27% even on best samples"*. Landing on the documented production floor from an independent implementation is a real cross-check on the method, not just on the result.

The per-tile shift also agrees with the synthetic nuclei-like estimate — real `fused/shim` median 1.085-1.105 and log-shift sd 0.061-0.066, against 1.067-1.071 and sd 0.053-0.063 synthetically. Section 2.3's choice of the nuclei-like row as the operative figure was right.

**Recommendation: ship the fused operator without recalibrating the cutoffs.**

What this does *not* establish: both samples pass with a wide margin, so neither exercises a sample sitting near 0.40, which is where a 1-2 pp shift would decide a verdict. Six 2048 px windows per sample is a spatial sample of the section, not the whole section. If a borderline sample is known, it is the one worth running — `tests/manual/real_calibration.py` takes a bundle's DAPI channel and prints the table above.

## 5. Reproducing the real-sample comparison

Paths come from the samplesheets of real Tower runs in `tower_launch/`:

```bash
# Xenium v1, real tissue (397 MiB DAPI channel)
aws s3 cp s3://altos-lab-genomics-data-spatialout/CI_TXG_XETG00378/\
20250612__132247__SPTL_009/output-XETG00378__0061499__R2_62985__20250612__132602/\
morphology_focus/morphology_focus_0000.ome.tif  v1_R2_control_dapi.ome.tif

# Xenium v2 / Atera preview (1.2 GiB DAPI channel, explicitly named)
aws s3 cp s3://altos-lab-bioinf/data/xenium_v2_preview_data/spatialraw/\
WTA_Preview_FFPE_Breast_Cancer/WTA_Preview_FFPE_Breast_Cancer/outs/\
morphology_focus/ch0000_dapi.ome.tif  atera_breast_dapi.ome.tif

CUDA_VISIBLE_DEVICES=0 python tests/manual/real_calibration.py
```

Two traps worth knowing before repeating this.

The v1 bundle names its channels positionally (`morphology_focus_0000.ome.tif`) while the v2/Atera bundle names them explicitly (`ch0000_dapi.ome.tif`); DAPI is channel 0 in both, but only one of them says so.

And a single downloaded channel file still carries the *multi-file* OME metadata for all four channels, so `tifffile` opens the OME series, fails to find the three siblings, and zero-fills them — at 4x the memory. Read `TiffFile.pages[0]` directly instead. `aszarr` is not an option either: `tifffile` 2025.9.20 requires zarr>=3 for that bridge, and `spatialqc` pins `zarr>=2.18,<3` deliberately.

## 6. What is still open

Section 1 of the plan is answered: two real samples, neither changing verdict, so the cutoffs stand.
What remains is narrower than "run real samples".

- **A borderline sample.** Both samples measured here pass with a wide margin (0.14 and 0.26 against `focus_warn: 0.40`). A sample already sitting near 0.40 is the only place a 1-2 pp shift decides anything, and none was available. If one is known, run it.
- **`pct_blurred_gmm_2d_roi_warn: 20.0`** (thresholds YAML line 338) is downstream of the same GMM fraction. `atera_breast` measures 26% blurry, which is already above that 20% warn line under *both* operators, so the fix does not change its verdict either — but that threshold is worth a look on its own merits, independently of this change.
- **`lap_focus_corr` on real data.** Section 2.5 measured deltas within +0.01 synthetically. Cheap to confirm on a real sample, and not expected to move.

The scripts are reproducible, and the first two take about a minute each on one L4:

```bash
# CPU-vs-GPU agreement, and fused-vs-shim divergence across lap_sigma
python tests/manual/log_gpu_check.py
# 3-seed GMM stability, blur-correlation sign, lap_focus_corr deltas
python tests/manual/gmm_invariance2.py
# real bundles: per-sample blurry fraction, fused vs shim, against the cutoffs
python tests/manual/real_calibration.py
```

## 7. Environment note for anyone reproducing this

`xenium-test-local` is missing `esda`, `libpysal` and `nsitk`.
The unit tests stub all three, so the suite passes without them, but a full end-to-end image QC run needs `nsitk` (`generate_tissue_mask` uses it on the path every run takes) and `esda`/`libpysal` (Moran's I for the negative-probe SNR metric).
It also carries `pyarrow 21.0.0`, which is above the cap anndata has previously required; that matters for transcript QC but not for the image QC paths exercised here.
