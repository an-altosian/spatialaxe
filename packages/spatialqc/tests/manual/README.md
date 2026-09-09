# Manual GPU validation scripts

These are **not** collected by pytest.
`testpaths` is `tests` and `python_files` is `test_*.py`, so these filenames are deliberately outside the pattern.

They require a real CUDA device and are meant to be run by hand when the GPU numerical paths change.
Findings from the first run are recorded in `docs/plans/2026-09-09_SPIKE_spatialqc-gpu-validation.md`.

## Why they exist

The package's CuPy branches were written without GPU access and merged unexecuted.
These scripts exercise the parts a unit test cannot assert cheaply: whether the CPU and GPU numerical paths agree, and whether a change to the Laplacian-of-Gaussian operator moves a sample-level QC verdict.

## Running them

```bash
export CUDA_VISIBLE_DEVICES=0
export NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/mpl   # both must be writable
python tests/manual/log_gpu_check.py
python tests/manual/gmm_invariance2.py
```

Each takes roughly a minute on one L4.

| Script                | Question it answers                                                                                   |
| --------------------- | ----------------------------------------------------------------------------------------------------- |
| `log_gpu_check.py`    | Do the CPU and GPU focus maps agree? How far does the removed shim diverge from the fused operator?    |
| `gmm_invariance2.py`  | Does the operator change move the sample-level blurry-tile % that `focus_warn`/`focus_fail` gate on?   |

## Interpreting `gmm_invariance2.py`

Three signals matter more than the absolute numbers, which come from synthetic mosaics:

- `spearman(blur_sigma, log_shift)` — the **sign**. Negative means the operator raises `lap_var` more on sharp tiles than blurry ones, widening blur/focus separation. That is the safe direction.
- `lap_focus_corr` delta — feeds `lap_focus_corr_warn: 0.50` / `lap_focus_corr_fail: 0.25`.
- blurry-fraction delta across seeds — a **consistent** sign would mean the cutoffs are uniformly mis-scaled and need a recalibration factor. An inconsistent sign means the 2D GMM is merely unstable on marginally bimodal samples, which is a different problem and is not fixed by moving a cutoff.

The GMM fits `[log1p(focus_score), log1p(lap_var)]` with `covariance_type="full"` and no feature scaling.
A uniform rescaling of `lap_var` would be a pure translation in log space and a full-covariance GMM is translation-equivariant, so it would be harmless.
The observed rescaling is not uniform, which is why the fraction can move at all.
