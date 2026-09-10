# spatialqc vs nf-xenium-processing dev HEAD — numerical equivalence

Goal: prove the `spatialqc` package produces exactly the same results as the image QC code at the HEAD of `nf-xenium-processing`'s `dev` branch.

## Reference version identification (do not skip this)

The phrase "dev HEAD" is ambiguous in this checkout and picking the wrong ref wastes a full run.

| Ref                          | Commit        | `image_qc.py` blob | lines     | `snr_metrics.py` blob | lines    |
| ---------------------------- | ------------- | ------------------ | --------- | --------------------- | -------- |
| local `dev` (STALE)          | `1d99d81`     | `6bfb47ad5159`     | 9645      | `60d5e27e59b6`        | 1306     |
| checked-out `HEAD`           | `648d404`     | `6bfb47ad5159`     | 9645      | `60d5e27e59b6`        | 1306     |
| **`origin/dev` (TRUE HEAD)** | **`cd87a5d`** | **`aeabf6daac1c`** | **14183** | **`a33fdf79f0bb`**    | **1892** |

Findings:

- The local `dev` ref is stale; `cd87a5d` is a descendant of it (`git merge-base --is-ancestor dev cd87a5d` returns true).
- `origin/dev` HEAD **is** `cd87a5d`, which is the exact commit spatialxe's QC was re-synced from (commit `eb56727`, "re-sync image QC from internal dev (5e35cae -> cd87a5d)").
- Therefore the port is a **refactor of the current dev HEAD**, not a port across a version gap. The streaming/GPU/tiled machinery, the consumer accumulators and the batched Otsu SNR all exist in the reference too.
- Using the stale 9645-line file as the reference is a trap: it makes the port look 4000 lines divergent and produces a meaningless comparison.

Authoritative reference paths at `origin/dev`:

```text
modules/local/image_qc/resources/usr/bin/image_qc.py     # blob aeabf6daac1c
modules/local/image_qc/resources/usr/bin/snr_metrics.py  # blob a33fdf79f0bb
```

Port under test:

```text
.claude/worktrees/spatialqc-package/packages/spatialqc/src/spatialqc/image/qc.py   # 13643 lines
.claude/worktrees/spatialqc-package/packages/spatialqc/src/spatialqc/image/snr.py
```

Commits on the port after the package move (`0e483ec`), all candidates for behavioral drift:

| Commit    | Subject                                                     | Numeric risk    |
| --------- | ----------------------------------------------------------- | --------------- |
| `9fc1982` | state the df_tx invariant mypy cannot infer                 | low             |
| `8159d0c` | replace the mypy assert with structure, drop dead guards    | low             |
| `9771f53` | bind each tile worker to its own card for the whole tile    | GPU-only        |
| `8e9cd23` | stop neg_pct emitting a divide warning on every real sample | **investigate** |

## Cleared divergences

- **`--stain-names`**. The _pipeline module_ stopped passing it (spatialxe commit `a5241ec`), but the **script keeps the flag on both sides** — it is in both click lists and both signatures (`ref:13093`, `qc.py:12448`), which is why the option counts are 20 vs 21 with `--device` as the only delta. It is numerically inert in both: the handling block is byte-identical (`ref:13190-13204` vs `qc.py:12546-12560`), and in each file `stain_names_list` is assigned, logged, then never read again. Neither run passes it, so it cannot affect this comparison either way.

- **Threshold configuration is not a variable here.** Neither run passes `--roi-thresholds-yaml`, and both logged `Using built-in default QC thresholds` with identical banners. The two repos could ship different `roi_image_qc_thresholds.yaml` files, so pinning one YAML would matter for a config-sensitive test; because both runs take the in-code defaults, and the code that defines them is in the byte-identical set, the measured diff is pure code.

### The vendored `xenium_helpers` surface — audited clean

The dev-HEAD reference imports exactly three helpers:

```python
from xenium_helpers.utils import (
    read_xenium_analysis_sw_version,
    read_xenium_major_version,
    resolve_segmentation_software,
)
```

The port does **not** depend on `xenium_helpers` at all; it vendored these into `spatialqc/bundle.py` ("only the surface was modernised").
This is a real divergence risk because `read_xenium_major_version` is numerically load-bearing — it selects XOA-version-specific QC intensity floors, which differ sharply between XOA 3.x and 4.0.

| Function                                         | Upstream       | Port            | Verdict           |
| ------------------------------------------------ | -------------- | --------------- | ----------------- |
| `read_xenium_analysis_sw_version`                | `utils.py:367` | `bundle.py:93`  | equivalent        |
| `_parse_xenium_version` / `parse_xenium_version` | `utils.py:386` | `bundle.py:137` | equivalent        |
| `read_xenium_major_version`                      | `utils.py:403` | `bundle.py:164` | equivalent        |
| `resolve_segmentation_software`                  | `utils.py:415` | `bundle.py:236` | equivalent        |
| `SEGMENTATION_PRETTY`                            | `utils.py:318` | `bundle.py:56`  | content-identical |
| `SEGMENTATION_TOOL_KEYS`                         | `utils.py:328` | `bundle.py:67`  | content-identical |

Notes on the two that are not textually identical:

- `read_xenium_analysis_sw_version`: the port factors the file read into `_load_experiment_json`, which keeps the same `is_file()` guard and the identical exception tuple `(OSError, ValueError, TypeError, json.JSONDecodeError)`. It adds one hardening — a top-level JSON value that is not a mapping returns `None`, where upstream would call `.get()` on a non-dict and raise an uncaught `AttributeError`. For a real bundle (a JSON object) the two are identical; the port is strictly safer.
- `parse_xenium_version`: upstream appends-then-breaks, the port breaks-then-appends, over the same `split("-", 1)[-1]` tail and the same `[:3]` truncation. Same output for every input.

### CLI contract — one delta, inert at its default

Every CLI option the two share has an **identical default**, checked by parsing the `@click.option` blocks out of both files (`scratchpad/cmp_opts.py`):

```text
--roi-size 35 | --max-scatter-points 10000 | --lap-sigma 1.0 | --max-gpus 0
--stream-tiles/--no-stream-tiles True | --figures/--no-figures True
--figure-source-tables False | --pipeline-segmentation "skip" | --is-resegmented False
--legacy-focus False | --no-snr False | --snr-no-roi-tx-table False
--snr-otsu-max-rois None | --snr-with-moran False | --save-dapi-maps-tiff False
--stain-names None | --roi-thresholds-yaml None | --sample-id None
```

The single difference is that the port **adds** `--device {auto,cpu,gpu}` with `default="auto"` (`qc.py:13620`).
`auto` means "use the GPU when one is present", which is the reference's implicit behaviour, so the flag is an opt-in override that is numerically inert at its default. `cpu`/`gpu` exist so an automated caller can fail loudly on a misconfigured GPU node instead of silently running ~50x slower.

Note the reference at dev HEAD already carries `--max-gpus`, `--pipeline-segmentation`, `--is-resegmented`, `--figure-source-tables` and `--figures/--no-figures`, which further confirms the two are the same generation.

### Commit `8e9cd23` — cleared

The one post-port commit flagged as a numeric-change suspect only suppresses a warning:

```python
# before
df["neg_pct"] = np.where(total_c > 0, neg_c / total_c, np.nan)
# after
df["neg_pct"] = np.divide(neg_c, total_c, out=np.full(n_rois, np.nan), where=total_c > 0)
```

`np.where` evaluates both branches, so the `0/0` for empty tiles was computed and discarded — correct result, but a `RuntimeWarning` on every sample. Both forms yield `neg_c/total_c` where `total_c > 0` and `NaN` elsewhere, so they are mathematically identical. The commit message additionally records an empirical re-run on this same bundle over 712,236 rows with identical `neg_pct`, `roi_tx_snr_ratio`, `snr_real_tx`/`snr_neg_tx`/`snr_total_tx`, `pct_blurred_gmm_tissue_filtered` (34.691941 unchanged) and `overall_snr_verdict` (WARN unchanged).

## Container / helper-version trap

`origin/dev`'s `conf/modules.config` pins `IMAGE_QC:ANALYSIS` to `xenium-processing-gpu-0.0.16` (ECR).
The Docker Hub image of the same tag, `docker.io/altoslabscom/xenium-processing-gpu:0.0.16`, carries an **older** baked `xenium_helpers`, so the dev-HEAD reference fails at import:

```text
ImportError: cannot import name 'read_xenium_analysis_sw_version' from 'xenium_helpers.utils'
```

The reference script supports an official override, which is the right fix:

```python
if "XENIUM_HELPERS_PATH" in os.environ:
    sys.path.insert(0, os.environ["XENIUM_HELPERS_PATH"])
```

This run mounts `bin/xenium_helpers/src/xenium_helpers` from `origin/dev` and prepends it via `PYTHONPATH`, which is equivalent (both precede `site-packages`), verified in-container:

```text
xenium_helpers resolved to: /helpers/xenium_helpers/utils.py
has read_xenium_analysis_sw_version: True
```

The port is unaffected by this trap precisely because it vendored the helpers.

## Determinism

Both implementations are deterministic, so a cross-run diff is meaningful:

- `GaussianMixture(..., random_state=random_state)` with `random_state: int = 0` default, in both.
- `np.random.default_rng(42)` for subsampling, in both.

## Method

Both scripts run in the **same container** (`docker.io/altoslabscom/xenium-processing-gpu:0.0.16`) on the **CPU path**, so any output difference is a code difference rather than a library or float-environment difference.

- Bundle: `/home/dhe/qc_e2e/bundle` (real Xenium, 34058x25623, 4 channels, JPEG2000-tiled).
- Thread counts pinned identically for both runs: `OMP/MKL/OPENBLAS/NUMBA_NUM_THREADS=16`.
- One heavy run at a time.
- Comparison by script, not by eye: `scratchpad/compare_outputs.py` compares file sets, every numeric JSON leaf exactly, table row/column parity plus per-column `max|delta|`, and figure dimensions. Volatile keys (timestamps, versions, paths) are reported separately, never silently dropped.

Sanity signal already observed: the reference CPU run reports `Tile grid: 712,236 tiles`, matching the independently measured `total_rois = 712236`.

## Host env trap — fixed (two layers)

`/home/dhe/.local/share/mamba/envs/xenium-test-local` could not run image QC at all: every attempt died ~2 s in at `generate_tissue_mask` (`qc.py:683`), where `nsitk.signed_maurer_distance_map` lazily imports napari.

**Layer 1 — napari/pydantic.** The host had **napari 0.6.6**, the working container **napari 0.7.0**, with _identical_ pydantic 2.12.5. So this was a napari version problem, not a pydantic pin:

```text
pydantic.v1.errors.ConfigError: duplicate validator function
  "napari.utils.theme.Theme._ensure_font_size"
```

Fixed by `pip install napari==0.7.0`. The upgrade is confined to the napari/GUI stack — `app-model`, `napari-plugin-engine`, `npe2`, `superqt`, `vispy`, plus new `pydantic-extra-types` / `pydantic-settings` / `python-dotenv`. **No numerical dependency moved**, verified before and after:

```text
numpy 2.2.6  scipy 1.16.2  skimage 0.25.2  pandas 2.3.2  sklearn 1.7.2
```

Rollback if ever needed:

```bash
pip install napari==0.6.6 app-model==0.4.0 napari-plugin-engine==0.2.0 \
            npe2==0.7.9 superqt==0.7.6 vispy==0.15.2
```

**Layer 2 — napari's theme cache (newly exposed by the fix).** With the `ConfigError` gone, the next error surfaced:

```text
OSError [Errno 30] Read-only file system:
  '/home/dhe/.cache-x86_64-glibc2.39/napari/.../_themes/dark/move_front_50.svg'
```

napari writes theme assets into the user cache, and `/home/dhe/.cache-x86_64-glibc2.39/` is not owned by this user. Fixed with `XDG_CACHE_HOME` pointed at a writable directory — the same class of fix the project already applies for `MPLCONFIGDIR` and `NUMBA_CACHE_DIR`. So a host run needs all three:

```bash
export MPLCONFIGDIR=<writable> NUMBA_CACHE_DIR=<writable> XDG_CACHE_HOME=<writable>
```

Verified end to end on the real bundle — the line that had never once appeared now does:

```text
[INFO] Generated tissue masks and distance maps
[INFO] [TIMING] Tissue mask generation: 13.7s
[INFO] No GPUs detected, using CPU backend
[INFO]   Tile grid: 712,236 tiles
```

A diagnostic aside worth remembering: `pip` reported `EROFS` twice for reasons that were **not** the filesystem. Once because pip defaults to a read-only `--user` site here (needs `PIP_USER=0`), and once because the Bash sandbox blocks writes outside its allowlist. `findmnt` shows the mount is `rw` and a direct `touch` succeeds, so `test -w` and `os.access(..., W_OK)` both report writable — they check permission bits, not the sandbox or the mount. Diagnose an `EROFS` from an actual write attempt, not from `access()`.

## Environment notes

- This host (`dockyard-main-0`) has **no GPU**: no `/dev/nvidia*`, no `nvidia-smi`, no `libcuda`; cupy imports but `getDeviceCount()` raises `cudaErrorInsufficientDriver`. Only the CPU path is testable here.
- The host env `/home/dhe/.local/share/mamba/envs/xenium-test-local` **cannot run the CPU path at all**: `generate_tissue_mask` (`qc.py:683`) calls `nsitk.signed_maurer_distance_map`, which lazily imports napari, which dies with `pydantic.v1.errors.ConfigError: duplicate validator function "napari.utils.theme.Theme._ensure_font_size"`. The container is unaffected.
- `docker logs` fails under the Bash sandbox (denied read of `~/.docker/config.json`); docker calls need the sandbox disabled.

## THE ANSWER — the 2x2 that settles it

Running both implementations on both backends isolates the anomaly to a single cell. Three of the four cells agree **exactly**; the outlier is always _reference-on-GPU_:

| Metric                                        | CPU reference      | CPU port           | **GPU reference**      | GPU port           |
| --------------------------------------------- | ------------------ | ------------------ | ---------------------- | ------------------ |
| `blur_gmm_2d.pct_blurred_gmm_tissue_filtered` | 34.69194064438745  | 34.69194064438745  | **35.03838550870625**  | 34.69194064438745  |
| `blur_gmm_2d.rois_blurred_gmm`                | 487910             | 487910             | **489100**             | 487910             |
| `laplacian_sharpness.lap_var_median_raw`      | 2066.907470703125  | 2066.907470703125  | **1926.1181640625**    | 2066.908203125     |
| `morphology.usable_tissue_frac`               | 0.6530136336243664 | 0.6530136336243664 | **0.6495550075839401** | 0.6530136336243664 |

Reading it:

- **The port matches dev HEAD's CPU answer exactly**, on both backends.
- **The port is backend-stable on every discrete decision — measured, not inferred.**
  An earlier draft of this section claimed the port's only CPU-vs-GPU movement was `lap_var_median_raw` at ~3.5e-07 and called it "self-consistent".
  That was asserted from four hand-picked keys, not measured, and it was wrong in detail.
  A full comparison of the port's own CPU and GPU outputs across the six shared metric JSONs gives:

| Class of key                                                  | Count | Result                                     |
| ------------------------------------------------------------- | ----- | ------------------------------------------ |
| Integer keys (ROI counts, cell counts, classification totals) | 236   | **236 exact, 0 differing**                 |
| Float keys, exactly equal                                     | 148   | —                                          |
| Float keys, differing                                         | 94    | max relative difference **3.68e-06**       |
| Differing floats above one float32 ULP (1.19e-07)             | 22    | all order statistics or large-N aggregates |

The four metrics in the 2x2 above are all in the exact set, including `blur_gmm_2d.rois_blurred_gmm` = 487910 and `morphology.usable_tissue_frac` = 0.6530136336243664 on both backends.
The largest movers are `focus_score.median` (3.68e-06), `focus_score_norm.min` (3.11e-06) and `median_blur_prob_gmm_2d_roi` (2.19e-06).
All three are order statistics over 712,236 ROIs, which are discontinuous in their inputs: a 1-ULP shift in individual focus scores selects a neighbouring element of a densely packed distribution, so the reported quantile moves by more than one ULP while nothing was actually reclassified.
The means and standard deviations sit at ~5e-07, consistent with pairwise-summation error over 712k values.
**The decisive point is that no float drift crossed a classification threshold: every count and every percentage-of-total is identical.**

- **Two configuration deltas surfaced in that same comparison, and neither is a code difference.**
  The local port run and the Tower port run disagreed on the threshold _inputs_ themselves — `dapi.critical_threshold` 500.0 vs 50.0, `intrna.critical_threshold` 300.0 vs 20.0, `pct_warn_threshold` 15.0 vs 40.0 — and on the 23 counts, percentages and statuses derived from them.
  These are values read from configuration, not measurements of the image, so they are classified separately and excluded from the numbers above (12 shared input keys plus the Moran set, whose keys differ in presence rather than in value).
  Critically, **the Tower-reference-vs-Tower-port comparisons show zero threshold differences**, so both Tower arms used identical thresholds and the GPU cell of the 2x2 is _not_ confounded by them.
  The second delta is the Moran flip described in Class 1 below.

- **The reference's two runs disagree with each other by ~7 % on `lap_var_median_raw`**, and that propagates into the 2-D GMM fit, the blur classification and `usable_tissue_frac`.
  This **is** a demonstrated property of the reference code: the shift reproduces bit-identically across two verified-different images, and its cause is the GPU/CPU operator mismatch identified in "ROOT CAUSE" below. The caveat that once stood here, that the two runs differed in image, has been discharged by measurement.
- So **spatialqc is not the source of any divergence found in this investigation.** Every disagreement traces to the single reference-on-GPU cell.

**The honest caveat on that cell, and on the comparison above.**
Reference-on-GPU ran in the ECR image while reference-on-CPU ran in the Docker Hub image with dev-HEAD's `xenium_helpers` mounted, so the reference's CPU-vs-GPU inconsistency is confounded with the image difference.
The port's own CPU-vs-GPU comparison carries the same class of confound — its CPU run was local in the Docker Hub image and its GPU run was on Tower — which is exactly what the threshold deltas above reveal.
So neither implementation has a _clean_ backend comparison, and this document does not claim one.
What the port's comparison does establish is a much weaker but still useful statement: across two different images, two different backends and two different threshold configurations, **every discrete output of the port was identical**, and the reference's ~7 % Laplacian shift has no counterpart of any size in the port.
Attributing the reference's shift to a defect in the reference would need the single combined image described below; until that exists, the shift is observed and unattributed, and nothing here identifies a reference-specific defect.
Neither result changes the finding about spatialqc, which rests on the controlled CPU test.

### `moran_p_sim` is not reproducible, in either implementation

Worth recording because it will otherwise be mistaken for a regression.
The reference's own `moran_p_sim` is 0.34 at 4 GPUs and 0.16 at 1 GPU, with `seed: 42` reported in both runs and `moran_i` identical to all printed digits (-0.00021797113894864514).
It is a permutation-test p-value on a null result — `moran_i` is ~-2e-04, i.e. no spatial autocorrelation — so the p-value is sampling noise on a uniform null and carries no information here.
The cause is **not isolated**: the two runs differed in GPU count, but an unseeded global RNG would explain the same observation equally well, and no two same-configuration runs were compared to tell them apart.
`moran_p_sim` must therefore be excluded from any exact-equality expectation, and the earlier statement that each side was "GPU-count-stable" holds for the substantive metrics but **not** for this key.

## Code-level audit result

A full symbol-by-symbol audit of `image_qc.py`@`cd87a5d` (14183 lines) vs `qc.py` (13643) and `snr_metrics.py` (1892) vs `snr.py` (1875) found **no difference that changes a number on the CPU path**.

- 102 of 114 paired symbols in the main module are byte-identical after docstring stripping; 17 of 37 in the SNR module, with the other 18 differing only in type annotations.
- Nothing present in the reference was dropped from either file.
- Output contract identical by mechanical set-diff, not inspection: output filenames (every `.json`/`.csv`/`.parquet`/`.png`/`.pdf`/`.tif`/`.gz`/`.h5` literal), dataframe column accessors, and the metric key sets of `save_roi_qc_metrics`, `save_cell_qc_metrics`, `save_simple_qc_metrics` and `compute_whole_grid_stain_percentiles`. Zero reference-only and zero port-only in all three categories.

Three items were flagged and then resolved:

| Area                                                                                | Change                                                                                                                                                                                                           | Verdict                                                                                                                                                                                                                                                                                                                                                                                                            |
| ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| YAML channel resolution (`qc.py:138` `_yaml_channel_cfg`, used at `:7099`/`:12935`) | Hand-written casing map `{'dapi':'DAPI',...}` replaced by a case-insensitive lookup that **raises `KeyError`** when `channels:` is populated but lacks the requested channel, instead of silently returning `{}` | **Inert for the shipped YAML.** `roi_image_qc_thresholds.yaml` defines exactly `DAPI:`, `boundary:`, `intRNA:`, and every call site passes those names, so all lookups resolve to the same sub-dict. Behaviour differs only on a _malformed_ YAML, where a silent fallback to `_INTENSITY_CRITICAL_DEFAULTS` / `intensity_warn=0.15` is now a hard failure. Absent or empty `channels:` still yields `{}` in both. |
| `--device` flag (`qc.py:953` `resolve_available_gpus`, option at `:13619`)          | Inline GPU selection extracted into a function with three modes                                                                                                                                                  | **Inert at the default.** At `device="auto"` it runs `detect_gpu_ids()` then the character-identical cap-and-slice. `cpu` and `gpu` are opt-in and unreachable without passing the flag.                                                                                                                                                                                                                           |
| GPU tile-worker device binding, commit `9771f53` (`qc.py:3472-3495`)                | Tile compute and the consumer fold now run inside `with cp.cuda.Device(gpu_id):`                                                                                                                                 | **GPU-only; not on the CPU path.** Same arithmetic, different card executes it. A no-op on a single GPU. On **multi-GPU this can change results** — in the direction of correctness, since without the context CuPy consumer ops could touch arrays owned by another device. Not testable on this host.                                                                                                            |

The last row is the only place where the port may legitimately _not_ match dev HEAD: a multi-GPU production run. That is a fix, not a regression, but it means "bit-identical to dev HEAD" cannot be claimed for multi-GPU without a GPU host to measure on.

Pairing was done on the AST with docstrings stripped recursively and re-emitted through `ast.unparse`, so comments, blank lines and quoting style cannot create false differences:

|                          | `image_qc.py` → `qc.py` | `snr_metrics.py` → `snr.py` |
| ------------------------ | ----------------------- | --------------------------- |
| byte-identical           | 102                     | 17                          |
| differing                | 12                      | 20                          |
| reference-only (dropped) | 0                       | 0                           |
| port-only (added)        | 3                       | 0                           |

The 3 port-only symbols are extractions of existing reference code, not new logic: `run_image_qc` (the body of the reference's `main`), `resolve_available_gpus`, and `_yaml_channel_cfg`. 18 of the 20 SNR differences are type annotations only (`Optional[X]`→`X | None`, `Dict`/`List`/`Tuple`→lowercase, `'Cls'`→`Cls`). Reference `main` vs port `run_image_qc` is 13 changed lines.

### Expected regression result: bit-identical, no tolerance

Because every function on the CPU path is in the byte-identical set — and on CPU both codebases really do call `scipy.gaussian_laplace`, so the operator swap does not apply — the correct expectation **for the CPU path** is exact equality — tile grid and tile coords, all focus/intensity/Laplacian columns, `focus_score_norm`, `roi_focus_score_threshold`, both GMM stages, tissue-mask stats, every SNR component, and the full contents of `roi_qc_metrics.json`, `cell_qc_metrics.json`, `roi_blur_threshold.json`, `snr_metrics.json` and `SNR_roi_tx.parquet`.

A float tolerance would be the wrong instrument here: any difference at all indicates a packaging defect — an import resolving to a different module, or a CLI/params wiring slip — and should be chased as such rather than absorbed. `compare_outputs.py` is therefore written for exact equality (with `NaN == NaN` treated as equal, since both mean "not computed").

That `Tile grid: 712,236` already matches on both sides is the _expected_ result rather than a coincidence: `compute_roi_grid` / `_build_roi_grid` are in the byte-identical set and the grid depends only on shape, `roi_size` and `stride`.

### Residual risks outside a CPU test

1. **Multi-GPU**: `9771f53`'s device binding means a multi-GPU reference run is not guaranteed to reproduce. Verify GPU-vs-CPU equivalence on the _port_; do not infer it from the reference.
2. **Malformed thresholds YAML**: `_yaml_channel_cfg` converts a silent misconfiguration into a hard `KeyError`. It never fires on the shipped YAML, but any deployment shipping a `channels:` block that omits one of the three channels used to fall back silently to `_INTENSITY_CRITICAL_DEFAULTS` with `intensity_warn=0.15` and will now abort. Worth checking any non-default thresholds YAML before rollout.

## Early empirical agreement

Both runs, in the same image on the CPU path, independently report the same grid before diverging into the expensive phase:

| Signal            | Reference (dev HEAD)                                                                                                          | Port (spatialqc) |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------- | ---------------- |
| backend selected  | `No GPUs detected, using CPU backend`                                                                                         | same             |
| `Tile grid`       | `712,236 tiles`                                                                                                               | `712,236 tiles`  |
| threshold banner  | `roi_intensity_threshold=100.0, min_tissue_cov=0.2, roi_focus_pct=5.0, blur_prob_thresh=0.5, ccfs_low_texture_threshold=0.02` | identical        |
| peak RSS observed | ~19-26 GB                                                                                                                     | ~19-26 GB        |

712,236 also matches the independently measured `total_rois` from the GPU baseline, so the ROI grid is consistent across reference, port, and the GPU streaming path.

## RESULT — CPU path: identical

Both runs completed with exit 0 in the same container image (reference 873 s, port 915 s).

| Check                    | Result                                                                                                                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| File set                 | **identical**, 54 files each                                                                                                                                                                                      |
| JSON metrics             | **no numeric differences in any shared JSON** — 6 files, 600 leaves, **505 numeric leaves**, all exactly equal                                                                                                    |
| Tables                   | **all 5 shared tables match exactly**, including `SNR_roi_tx.parquet` at 712,236 rows                                                                                                                             |
| Figures                  | 40 shared figures, **0** with differing dimensions                                                                                                                                                                |
| Volatile-key differences | **none** — the JSONs contain no wall-clock fields, so the quiet volatile section is real rather than a blind spot (only version-ish key is `xoa_version`, a data field read from the bundle, equal on both sides) |

Agreeing values on both sides:

```text
total_rois                            712236
total_rois_tissue_filtered            343489
total_cells                           126045
overall_snr_verdict                   WARN
xoa_version                           xenium-3.3.0.1
blur_gmm_1d.rois_blurred_gmm          489659
blur_gmm_1d.pct_blurred_gmm_tissue_filtered   35.22849348887446
```

`xoa_version = xenium-3.3.0.1` means `read_xenium_major_version` returned 3 and the XOA 3.x intensity floors were selected, so the vendored helper audited above was genuinely exercised on this run, and both sides agree.

**The compare script exits 1, and that is not a ref-vs-port failure.** It is the _sanity_ assertions, which are hard-coded to the peer's **GPU** baseline. Two of them differ from CPU — `rois_blurred_gmm` 489659 vs 487910, and `pct_blurred_gmm_tissue_filtered` 35.2285 vs 34.6919 — and they differ **identically on both sides**, so this is a CPU-vs-GPU difference, not a port defect. That is expected: the CPU branch is a different algorithm from the GPU streaming path (whole-image `compute_ccfs_map` and a scalar per-ROI SNR loop instead of tiled streaming with a batched device Otsu), and the 1-D blur GMM is fit on focus scores, so small float differences in the focus map shift a threshold-based count by ~0.36 %. The `image_qc_metrics.json` "51 vs 40 keys" line is likewise a counting-convention difference in my flattener, not a contract change — the file-set and key-set diffs are both empty.

## GPU verification on Seqera Platform (in flight)

The CPU result cannot speak to the one open item — commit `9771f53`'s multi-GPU device binding. Two arms were launched on Tower against the same bundle, pinned to exact commits verified against GitHub with `git ls-remote`:

| Arm           | Pipeline @ commit                                       | Params                                         | Workflow ID      |
| ------------- | ------------------------------------------------------- | ---------------------------------------------- | ---------------- |
| A — reference | `altos-labs/nf-xenium-processing` @ `cd87a5d9397939...` | `segmentation: skip`, `image_qc_gpus: 4`       | `pmZm9OvOTrFTF`  |
| B — port      | `altos-labs/spatialxe` @ `882ed8a6616aaa...`            | `mode: qc`, `run_qc: true`, `image_qc_gpus: 4` | `3Uc5VRLlayQCE2` |

Both on compute env `4WpM1mBmA2qSC6CxukQ1yx` (`GPU_Fusion_NS_09Feb_onDemand_spatial`), profile `gpu,aws`, workspace `26707052208082`.

Two traps found while launching:

- The compute env IDs documented in `scripts/tower-launch.sh` are **stale**. `5ixknRlHbJM1T0Fr2iRPW7` ("GPU big") is **deleted** — the API returns `HTTP 400 {"message":"The selected compute environment (deleted-33747942085129513 - 5ixknRlHbJM1T0Fr2iRPW7) is not available any more"}`. The script's error handler swallows the response body, so this surfaces only as a bare `HTTP 400`; POST directly to see the message. Worse, `nf-xenium-processing`'s `conf/base.config` pins `process_gpu_qc` to `queue = 'TowerForge-5ixknRlHbJM1T0Fr2iRPW7'`, i.e. that dead env — survivable only because the launch script's `configText` overrides `queue` from the chosen compute env.
- The two pipelines take **different samplesheet formats**: spatialxe wants `sample,bundle,image`; nf-xenium-processing wants `id,xenium_bundle,he_path,ref_id,ref_h5ad_path,ref_h5ad_cols,ref_query,min_mols_per_cell,min_genes_per_cell`. Two samplesheets are therefore needed for one bundle: `spatialqc_verify_1sample.csv` and `ref_verify_1sample.csv`, both under `s3://altos-lab-bioinf/SDI/pipelines/spatialxe/samplesheets/`.

`image_qc_gpus: 4` is a request, not a guarantee — the env is Forge-managed and its allowed instance types are not exposed via the API. Read the actual device count from each run's IMAGE_QC log before claiming anything about multi-GPU; if Forge provisions a single-GPU instance, this closes single-GPU parity only and the `9771f53` caveat stays open.

## RESULT — 4-GPU path: NOT identical

Both arms SUCCEEDED on Tower (`2Mce5R6pieSt8B` reference, `2ewDL90CwV58z9` port), both on **g6.24xlarge = 4x NVIDIA L4**, so `image_qc_gpus: 4` was honoured and the multi-GPU path really was exercised. Reference took 364 s, port 584 s.

Confounds ruled out first:

- **Thresholds YAML is not a variable.** Both arms were passed `--roi-thresholds-yaml roi_image_qc_thresholds.yaml`, and the two repos' copies are **byte-identical** (both md5 `7a92c7ef774789e7099b6ab42f8c9e58`, 344 lines).
- Same bundle, same `--roi-size 35`, same `--max-gpus 4`, same instance type.

What still matches exactly: `total_rois` 712236, `total_cells` 126045, `total_rois_tissue_filtered` 343489, the whole **1-D** GMM chain (`rois_blurred_gmm` 489659, `pct_blurred_gmm_tissue_filtered` 35.22849348887446), `overall_snr_verdict` WARN, and `SNR_roi_tx.parquet` (matches exactly).

### Class 1 — Moran's I: a pipeline default flip, not a code difference

Same parameter name, **opposite default**:

| Repo                                           | `image_qc_snr_no_moran` | Effect            |
| ---------------------------------------------- | ----------------------- | ----------------- |
| `nf-xenium-processing` (`nextflow.config:176`) | `false`                 | Moran's I **ON**  |
| `spatialxe` (`nextflow.config:127`)            | `true`                  | Moran's I **OFF** |

So the reference was invoked with `--snr-with-moran` and the port was not, giving:

```text
ref  : snr.components.SNR_roi_neg_spatial.moran_i        = -0.00021797113894864514
       ...moran_p_sim = 0.34, moran_verdict = 'PASS', moran_subsample.{n_available,n_used,seed}
port : snr.components.SNR_roi_neg_spatial.moran_note     = 'skipped (Moran disabled; quadrant only)'
```

This is a **pipeline-level** divergence, not a spatialqc-package one — the script's own `--snr-with-moran` default is `False` on both sides (see the CLI table above). It is a real answer-changing difference for anyone running the pipelines with defaults, and it needs an explicit decision: either flip spatialxe's default back to Moran-on, or accept the change deliberately.

### Class 2 — the 2-D GMM / Laplacian family genuinely diverges

These are **not** float noise:

| Metric                                         | Reference (dev HEAD) | Port                  | Note                                     |
| ---------------------------------------------- | -------------------- | --------------------- | ---------------------------------------- |
| `laplacian_sharpness.lap_var_median_raw`       | 1926.1181640625      | 2066.908203125        | **~7 % apart**                           |
| `blur_gmm_2d.pct_blurred_gmm_tissue_filtered`  | 35.03838550870625    | **34.69194064438745** | port equals the established GPU baseline |
| `blur_gmm_2d.rois_blurred_gmm`                 | 489100               | 487910                | 1190 ROIs                                |
| `blur_gmm_2d.rois_blurred_gmm_tissue_filtered` | 120353               | 119163                |                                          |
| `gmm_2d.component_means[1][1]`                 | 2.991532717765199    | 3.1154281320721466    |                                          |
| `gmm_2d.component_weights[0]`                  | 0.6508443887780668   | 0.6551936564342157    |                                          |
| `morphology.usable_tissue_frac`                | 0.6495550075839401   | 0.6530136336243664    |                                          |
| `laplacian_sharpness.focus_lap_spearman_corr`  | 0.8334719347367797   | 0.832564757910699     |                                          |
| `cells_blurred_gmm_2d_roi`                     | 2280                 | 2114                  | propagates to every `cluster_blur.*`     |

The shape of this is informative: the **1-D** chain (focus scores only) is bit-identical, while everything downstream of the **Laplacian variance map** and the **2-D** GMM moves. `lap_var_median_raw` shifting 7 % is the upstream cause, and the 2-D GMM, its component fit, the cell-level blur calls and `cluster_blur.*` are all consequences.

This is precisely what the code audit flagged in advance for commit **`9771f53`** — tile compute and the consumer fold now run inside `with cp.cuda.Device(gpu_id)`, which is a no-op on one card but on multi-GPU stops CuPy consumer operations from touching arrays owned by another device. The audit's verdict was "it CAN change GPU numbers, toward correctness"; that appears to be exactly what is being observed, with the port landing on the known-good baseline and the reference not.

### The multi-GPU attribution was WRONG — corrected

The 1-GPU isolation test (`1dvzIdir8LJ1Sz` reference, `4GMMVr85SABJV6` port, Moran aligned ON) **refutes** the `9771f53` hypothesis. Every difference is _identical_ at 1 GPU and at 4 GPUs, to the last digit:

```text
cells_blurred_gmm_2d_roi                     ref 2284              port 2118
gmm_2d.component_means[1][1]                 ref 2.991532717765199 port 3.1154281320721466
blur_gmm_2d.pct_blurred_gmm_tissue_filtered  ref 35.03838550870625 port 34.69194064438745
blur_gmm_2d.rois_blurred_gmm                 ref 489100            port 487910
laplacian_sharpness.lap_var_median_raw       ref 1926.1181640625   port 2066.908203125
```

And each side is individually **GPU-count-stable** — ref@4gpu == ref@1gpu and port@4gpu == port@1gpu for every metric checked, including `lap_var_median_raw`. So the device-binding change is genuinely inert on this sample, in both directions. The audit was right that it _could_ matter on multi-GPU; it does not matter here, and attributing this divergence to it was a mistake.

Two things the 1-GPU run did settle:

- **Moran alignment worked.** Setting `image_qc_snr_no_moran: false` on the port removed all seven Moran key differences, confirming Class 1 was purely the default flip. One residual: `moran_p_sim` ref 0.16 vs port 0.24 — a permutation p-value, downstream of the same divergence.
- The divergence is confined to the **Laplacian variance map** and everything downstream of it. The 1-D chain, `total_rois`, `total_cells` and `SNR_roi_tx.parquet` stay exact.

### The actual leading candidate: the arms ran in different images

The two arms used tag `0.0.16` from **different registries**:

```text
ref : wave.seqera.io/wt/58b4b69e7f4b/nextflow/containers:xenium-processing-gpu-0.0.16   (ECR)
port: wave.seqera.io/wt/db60697dfe95/altoslabscom/xenium-processing-gpu:0.0.16          (Docker Hub)
```

Those two are **provably not the same image**: the Docker Hub copy carries an older `xenium_helpers` that lacks `read_xenium_analysis_sw_version`, which is exactly why the dev-HEAD reference `ImportError`ed in the local container test above. Same tag, different contents — so the Tower comparison was never controlled for image, whereas the CPU comparison (both scripts in one image) was, and that one came out exactly identical.

That makes a library-version difference — most plausibly in the CuPy/CUDA stack that computes the Laplacian — the leading explanation for a 7 % shift in `lap_var_median_raw` appearing only on the Tower runs.

### PROVEN: the two `0.0.16` images are different software

The controlled test (`3NFtjYuS4FgqOr`, reference forced into the port's Docker Hub image) failed exactly as feared, and in doing so proved the point on Tower rather than only locally:

```text
ImportError: cannot import name 'read_xenium_analysis_sw_version'
  from 'xenium_helpers.utils' (/opt/conda/lib/python3.11/site-packages/xenium_helpers/utils.py)
```

So `docker.io/altoslabscom/xenium-processing-gpu:0.0.16` and the ECR `nextflow/containers:xenium-processing-gpu-0.0.16` **carry different code under the same version tag**. Two pipelines that are required to produce identical numbers were not running the same software, which invalidates the GPU A/B as a code comparison.

This is a finding to act on regardless of how the numbers resolve: pin both pipelines to one immutable image reference (ideally by digest, not tag).

### BLOCKED: the two images are mutually exclusive, so no controlled GPU test is possible

> **Superseded.** This was resolved by building a combined image — see "RESOLVED: the combined image exists" below. The analysis is kept because it is what motivated the build.

Both directions were attempted and both are impossible with the images as they exist:

| Attempt                                      | Run              | Outcome                                                                                                                                                                                                                    |
| -------------------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reference in the port's **Docker Hub** image | `3NFtjYuS4FgqOr` | `ImportError: cannot import name 'read_xenium_analysis_sw_version' from 'xenium_helpers.utils'` — that image's `xenium_helpers` is too old for dev HEAD                                                                    |
| Port in the reference's **ECR** image        | `2uBAuvJmm8udxy` | `Unschedulable ... JOB_RESOURCE_REQUIREMENT` — my own error: a `withName` selector **replaces** the label-derived directives, so `cpus`/`memory`/`accelerator` were lost and it requested the 94-vCPU `resourceLimits` cap |
| Port in the ECR image, resources restated    | `zpL2JTqQkg2jw`  | `spatialqc-image-qc: command not found` (exit 127) — the ECR image has no spatialqc wheel                                                                                                                                  |

Verified directly: `docker run docker.io/altoslabscom/xenium-processing-gpu:0.0.16 command -v spatialqc-image-qc` returns `/opt/conda/bin/spatialqc-image-qc`, while the ECR image has no such entrypoint.

So:

- the **ECR** image can run the reference but not the port (no spatialqc wheel);
- the **Docker Hub** image can run the port but not the reference (stale `xenium_helpers`).

**There is no image in which both can run**, and therefore the GPU comparison cannot be made controlled today. The CPU comparison was only controllable because it ran the two _scripts_ in one image with the reference's own `xenium_helpers` mounted and `PYTHONPATH`-prepended — a trick unavailable through a pipeline launch.

**To settle GPU equivalence, someone must first build a single image carrying both the spatialqc wheel and dev-HEAD's `xenium_helpers`**, then re-run this A/B. Until then the GPU divergence is unattributed: the code audit says it cannot be code, and the only remaining variable is the environment, but that is inference rather than measurement.

The `withName`-replaces-`withLabel` lesson is worth keeping: when overriding a container or queue for one process, restate `cpus`, `memory`, `time` and `accelerator` in the same block, or the task silently loses its resource request. Losing `accelerator` in particular would downgrade a GPU run to CPU without any error.

### TRAP: a failed QC analysis is reported as a SUCCEEDED workflow

`3NFtjYuS4FgqOr` reported workflow status **SUCCEEDED** while producing **no metrics at all** — only:

```json
{ "status": "failed", "exit_code": 1, "sample_id": "XETG00378__0061499__R2_control" }
```

The module wrapper deliberately converts the script's exit 1 into a written `image_qc_status.json` plus exit 0, so the Quarto report can render a QC-FAILED banner. That is reasonable by design, but it means **workflow status is not evidence that QC ran**. Anyone comparing outputs must check `image_qc_status.json`, or a run that produced nothing looks like a success.

Verified for this investigation — all four arms whose numbers are quoted above report `{"status": "ok"}`:

| Arm                                 | `image_qc_status.json`            |
| ----------------------------------- | --------------------------------- |
| 4-GPU reference                     | `ok`                              |
| 4-GPU port                          | `ok`                              |
| 1-GPU reference                     | `ok`                              |
| 1-GPU port                          | `ok`                              |
| 1-GPU reference in Docker Hub image | **`failed`, exit 1** (no metrics) |

So the comparisons above stand, and the controlled test does not.

## RESOLVED: the combined image exists, and it makes the CPU test airtight

The blocker described above is gone. One image now runs both scripts:

```text
altoslabscom/xenium-processing-gpu:0.0.16-equiv1
  base      docker.io/altoslabscom/xenium-processing-gpu:0.0.16
  + xenium_helpers  nf-xenium-processing origin/dev cd87a5d
  + spatialqc       spatialxe feat/spatialqc-package 882ed8a
  installed with: pip install --no-deps --force-reinstall
```

A correction to what this document said earlier: the Docker Hub image was described as lacking only up-to-date helpers, and the ECR image as lacking the spatialqc wheel.
Inspecting the Docker Hub image directly shows it already carries `spatialqc` 0.1.0 and the `spatialqc-image-qc` entry point.
So the single real gap was the stale `xenium_helpers`, and the fix is one layer, not a rebuild.

Both installs are `--no-deps`, and that is the whole reason the image is valid rather than merely functional.
`spatialqc`'s own metadata asks for `numpy>=2.0`, `pandas>=2.2`, `pyarrow>=18` — all satisfiable by _newer_ wheels — so an ordinary `pip install` would have been free to upgrade `scipy` or `scikit-learn` and silently invalidate every number the image exists to compare.
`--force-reinstall` covers the mirror-image failure: the port and the baked wheel both report version `0.1.0` while having different content, so a version-based decision would have skipped the replacement and tested the stale code.
The numeric stack was verified byte-identical to the base after building:

| Library      | Base and combined image |
| ------------ | ----------------------- |
| numpy        | 2.2.6                   |
| scipy        | 1.16.2                  |
| scikit-learn | 1.7.2                   |
| pandas       | 2.3.2                   |
| pyarrow      | 21.0.0                  |
| tifffile     | 2025.9.20               |
| numba        | 0.62.0                  |

The installed `spatialqc/image/qc.py` md5 matches the source under test (`5931f2b984855e932ee2ebde7309dc3e`), and the reference scripts were taken from the git object store, so their blob ids are provably dev HEAD: `aeabf6daac1c` (`image_qc.py`, 14183 lines) and `a33fdf79f0bb` (`snr_metrics.py`, 1892 lines).

### The mutual-exclusivity claim, demonstrated and then removed

| Image                 | Reference `image_qc.py`                                             | Port `spatialqc-image-qc` |
| --------------------- | ------------------------------------------------------------------- | ------------------------- |
| `0.0.16` (Docker Hub) | `ImportError: cannot import name 'read_xenium_analysis_sw_version'` | works                     |
| `0.0.16-equiv1`       | works                                                               | works                     |

### RESULT — single-image A/B: bit-identical

Run on the `Xenium_Prime_Mouse_Ileum_tiny_outs` bundle (`analysis_sw_version: xenium-3.0.0.15`, so major version 3 and the XOA-version intensity-floor path is genuinely exercised), both arms in one container, identical pinned threading, neither arm shadowed by a `PYTHONPATH`:

| Check                                          | Result                               |
| ---------------------------------------------- | ------------------------------------ |
| File sets                                      | identical, 39 files each             |
| JSON numeric leaves                            | **no differences at all**            |
| Integer keys (counts, totals, classifications) | 125 compared, 0 moved                |
| Float keys                                     | 187 exact, **0 differing**           |
| Tables                                         | all 5 match exactly                  |
| Figures                                        | 25 shared, 0 differing in dimensions |
| Exit codes                                     | `A_EXIT=0` (22 s), `B_EXIT=0` (25 s) |

This is stronger than the earlier CPU test in the one way that mattered.
There, each arm needed a `PYTHONPATH` to shadow something stale — the reference to reach dev-HEAD's helpers, the port to reach the code under test — so the two arms were not quite running in the same environment.
Here both are installed normally and the only variable left is the script.
The result is not "agrees to float tolerance"; it is zero differing values.

**Honest limits of this particular run.** The tiny bundle has 20 cells and 0 tissue-filtered ROIs, so `blur_gmm_1d.total_rois_tissue_filtered` is 0 and the tissue-filtered blur-GMM branch is degenerate, returning NaN where R2_control returns populated statistics.
Exactness on a degenerate branch is still exactness, but the populated branches are covered by the R2_control CPU run (505 numeric leaves), not by this one.
Together the two cover both regimes; neither covers both alone.

### Two harness bugs this run exposed, both fixed

1. **Both scripts need a writable working directory.** Arm A first died with `OSError: [Errno 30] Read-only file system: './analysis'`: the reference unpacks the bundle's `analysis.tar.gz` into its _current_ directory (`image_qc.py:13706`), so a read-only cwd kills it. An earlier draft of this section added "the port does not do this", which was wrong and was never tested — arm B's cwd happened to be `/tmp`, which is writable. The port unpacks into cwd in exactly the same way (`qc.py:13041`, `extract_dir="."`), so the constraint applies equally to both and is one more respect in which they behave identically. It is a harness constraint rather than a numeric difference, but it is worth knowing before mounting anything read-only, and it is why both arms now get their own writable cwd.
2. **`compare_outputs.py` hardcoded one bundle's counts.** Its sanity assertions asserted `total_rois == 712236` and `total_cells == 126045`, R2\*control's values, so the tiny-bundle run reported two scary `*** MISMATCH` lines and an overall "NOT numerically identical" verdict while both sides in fact agreed on every number. The constants were checking \_which bundle you ran\*, not whether the comparison was valid. They are replaced by the invariant that actually matters: the key exists, the value is non-trivial (so two empty runs cannot pass by agreeing on nothing), and the two sides agree. Re-running the original R2_control CPU pair through the fixed tool still returns "numerically identical", so the change did not weaken the earlier verdict.

A third bug was found in `residual_deltas.py`: `nan != nan` meant NaN-on-both-sides was counted as a difference, which reported 37 spurious diffs and a `nan` maximum on the tiny bundle. NaN on both sides is now agreement (matching `compare_outputs.py`), while NaN on one side only is a hard failure. With that fixed the single-image A/B shows 0 differing floats, and the R2_control CPU-vs-GPU maximum is unchanged at 3.676e-06.

### The GPU A/B — pushed, pinned by digest, launched

The image is on Docker Hub as a **new tag**, `0.0.16-equiv1`, which does not touch `0.0.16`, so nothing already deployed changed:

```text
docker.io/altoslabscom/xenium-processing-gpu:0.0.16-equiv1
digest sha256:2d1b38ae1a4a412da05f6f8466c1b98225a844a2d81cfc801f15b0ba56e76b3a
```

Both payloads reference the image **by digest, not by tag**. This investigation lost time to two different images sharing the tag `0.0.16`, so the arms of a comparison must point at content that cannot be re-pointed underneath them. This is recommendation 5 applied to itself.

Both launch payloads are written and validated in `scripts/qc-equivalence/`:

| Payload                            | Pipeline               | Revision  |
| ---------------------------------- | ---------------------- | --------- |
| `payload_equiv_ref_combined.json`  | `nf-xenium-processing` | `cd87a5d` |
| `payload_equiv_port_combined.json` | `spatialxe`            | `882ed8a` |

Both pin 1 GPU, both set `image_qc_snr_no_moran: false`, and both run the same combined image, so the script is the only variable. 1 GPU is deliberate: the divergence reproduced identically at 1 and 4 GPUs, and it makes `_compute_channel_maps_tiled` (`9771f53`) a no-op, removing the one substantive code delta from the comparison.

Two traps are pre-empted in these payloads:

- **The container is set inside `withLabel:process_gpu_qc`, not `withName`.** A `withName` selector _replaces_ the label-derived directives instead of merging with them. That is what silently dropped `cpus`, `memory` and `accelerator` from the last port launch and left it requesting 94 vCPU and no GPU, failing as `Unschedulable ... JOB_RESOURCE_REQUIREMENT`. Setting the container on the label the process already carries keeps every directive in one visible block.
- **`image_qc_snr_no_moran` is stated on both arms** rather than left to each repo's default, which still differ at these two revisions. Leaving it implicit would make the setting the variable under test instead of the code.

Launched:

| Arm | Pipeline               | Revision  | Workflow ID      |
| --- | ---------------------- | --------- | ---------------- |
| A   | `nf-xenium-processing` | `cd87a5d` | `4dvUPGV7n7tp3m` |
| B   | `spatialxe`            | `882ed8a` | `34euDFBKgP0LhS` |

Compare with:

```bash
python3 scripts/qc-equivalence/compare_outputs.py  <ref-outdir> <port-outdir>
python3 scripts/qc-equivalence/residual_deltas.py  <ref-outdir> <port-outdir>
```

The prediction on record, from the code audit plus the bit-identical CPU result: they will agree. If they do not, the cause is in the GPU path and the environment is no longer available as an explanation, which is exactly what this image was built to establish.

## WITHDRAWN: my "the image is not the cause" conclusion was based on a run that never used the image

This section previously claimed the environment had been refuted as the cause of the reference's cross-backend shift, because arm A in the "combined image" reproduced the old ECR numbers to every digit.
**That conclusion is withdrawn. The premise was false.**

Arm A (`4dvUPGV7n7tp3m`) did not run in the combined image. Its task record shows:

```text
container: wave.seqera.io/wt/9a866b4225a6/nextflow/containers:xenium-processing-gpu-0.0.16
```

That is the ECR image. So "the combined image reproduces the ECR numbers" was really "the ECR image reproduces the ECR numbers", which establishes nothing at all.

### Why the override silently failed

The container was set inside `withLabel:process_gpu_qc`, deliberately, to avoid the `withName`-replaces-label trap that had previously stripped `cpus`, `memory` and `accelerator` from a launch.
But the reference sets its own container at `conf/modules.config:114`:

```groovy
withName: '.*IMAGE_QC:ANALYSIS' {
    ext.prefix = "image_qc"
    container = "873817298425.dkr.ecr.us-west-2.amazonaws.com/nextflow/containers:xenium-processing-gpu-0.0.16"
}
```

`withName` outranks `withLabel`, so the ECR container won and my override was inert.
This is the same precedence rule that bit twice before, now biting from the opposite direction: avoiding `withName` to protect the resource directives is exactly what forfeited the container override.

**The correct override must use `withName` and restate everything**, because `withName` replaces the label-derived directives rather than merging with them — `cpus`, `memory`, `time`, `queue`, `accelerator`, and `ext.prefix` (which names the output directory, so losing it would move the results). Relaunched as `1vzu3lIL2FIwrV`.

### The lesson worth keeping

A container override is not verifiable from the config you submitted; only the task record shows what actually ran.
`GET /workflow/{id}/tasks` returns a `container` field per task, and **that field is the only proof**. It should be checked before any number from a run is interpreted, the same way `image_qc_status.json` is checked before a SUCCEEDED workflow is believed. Both failures are silent and both produce plausible-looking output.

## Where the CPU-vs-GPU discrepancy actually sits (cause not yet established)

What _is_ solidly measured, from the arm A / arm B comparison (both on `g6.16xlarge`, both GPU-resident, 160 s and 149 s of compute, so neither fell back to CPU):

| Output                                                                | Rows differing  | Max abs delta                |
| --------------------------------------------------------------------- | --------------- | ---------------------------- |
| `SNR_roi_tx.parquet` / `grid_roi_focus_scores.csv` col `dapi_lap_var` | 602504 / 712236 | 28709.4                      |
| col `blur_prob_gmm_2d`                                                | 311581 / 712236 | 0.38566                      |
| col `is_blurred_gmm_2d`                                               | 1206 / 712236   | 1 (genuine reclassification) |
| `image_qc_cell_metrics.csv` col `blur_prob_gmm_2d_roi`                | 126026 / 126045 | 0.38566                      |

The chain is unambiguous: **`dapi_lap_var` is the origin**, and everything else is downstream of it — the 2-D GMM is fitted on the Laplacian variance, so a shifted input moves the component means, the blur probabilities, the classification counts and `usable_tissue_frac` in turn.
`lap_var_median_raw` 1926.1181640625 vs 2066.908203125 is that shift expressed as one number.

A delta of 28709 on a variance, across 85 % of ROIs, is not float reduction order. It is a different computation.

**The leading hypothesis, not yet tested:** the CPU path computes the Laplacian over the whole image (`scipy`), while the GPU path computes it per tile through `cupyx` and stitches the results. A tiled convolution needs a halo of at least the kernel radius on every edge, or each tile boundary produces an incorrect gradient. That would corrupt a large fraction of ROIs — the ones near any tile edge — while leaving interior ROIs correct, which is the shape of what is observed (85 % differing rather than 100 %, and a large maximum rather than a uniform offset).

**Why this remains a hypothesis:** the arms still differed in image, so a code cause and an environment cause are not yet separated. The relaunched arm A settles that. Until it lands, the honest statement is that the discrepancy originates in `dapi_lap_var`, and its cause is unattributed.

The two candidate explanations, to be distinguished by that run:

1. The reference's GPU Laplacian differs from its own CPU Laplacian (halo/tiling arithmetic), and the port does not reproduce that behaviour.
2. The two images compute it differently — different `cupy`/CUDA versions, which nothing in `versions.yml` currently records. This is why recommendation 4 exists.

## ROOT CAUSE: the reference applies a different Laplacian operator on GPU than on CPU

The CPU-vs-GPU discrepancy is fully explained, and it is neither the container nor float reduction order. It is two different discrete operators.

At `image_qc.py:81-85` the reference defines its own GPU Laplacian-of-Gaussian:

```python
def cupy_gaussian_laplace(image, sigma):
    """CuPy Laplacian of Gaussian: Gaussian smooth then Laplacian."""
    from cupyx.scipy.ndimage import gaussian_filter as _gf
    return cupy_laplace(_gf(image, sigma=sigma))
```

Its CPU branch uses `scipy.ndimage.gaussian_laplace`, which is **not** the same thing.
`gaussian_laplace` applies an analytic Laplacian-of-Gaussian — a fused Gaussian _second-derivative_ kernel along each axis.
The shim applies a discrete 5-point Laplacian to an already-smoothed image.
Those two have different frequency responses, so the same input yields a different response depending only on which backend ran.

The port replaced the shim with the operator that was there all along (`qc.py:78`):

```python
from cupyx.scipy.ndimage import gaussian_laplace as cupy_gaussian_laplace
```

The port's own comment records why, and states that the shim was "added here on the stated grounds that cupyx exposes no `gaussian_laplace`. It does."

### Verified independently

The two operators were compared directly on a synthetic multi-scale field at the production `lap_sigma` of 1.0, on CPU, with no GPU involved — `scipy.ndimage.gaussian_laplace` versus `laplace(gaussian_filter(...))`:

| Quantity                              | Measured here | The port's recorded figure |
| ------------------------------------- | ------------- | -------------------------- |
| Correlation between the two responses | 0.9931        | 0.9937                     |
| Variance ratio, shim / fused          | 0.8096        | 0.77                       |
| Max relative deviation                | 0.1255        | 0.166                      |

The two sets of figures agree in direction and magnitude; they differ in detail because they were measured on different images, and the exact ratio depends on the spatial-frequency content of the input.

**Why this produces exactly the observed symptom.** The focus score is a _variance-like reduction_ of the Laplacian response. The shim's response has ~0.8x the variance of the fused operator's, so every ROI is graded as less sharp on GPU than the identical ROI on CPU. Reducing a windowed Laplacian variance through both operators on the synthetic field gives a median ratio of 0.807, in the same direction as the production ratio of 0.932 (`1926.118 / 2066.908`).

That explains the whole cascade, which is why `dapi_lap_var` was the origin and everything else moved downstream of it:

```text
different LoG operator on GPU
  -> dapi_lap_var shifted low          (602504 / 712236 ROIs differ, max delta 28709)
  -> 2-D GMM fitted on shifted input   (component means and weights move)
  -> blur probabilities shift          (311581 / 712236 ROIs, max delta 0.386)
  -> ROIs reclassified                 (1206 / 712236 flip is_blurred_gmm_2d)
  -> usable_tissue_frac, pct_blurred, gmm_comparison counts all move
```

### The consequence the reference has and the port does not

The port's comment names a failure mode worse than the cross-backend gap itself: _"the GPU-OOM fallback at `compute_all_focus_maps` switched operators part-way through a sample."_
In the reference, a GPU OOM mid-sample falls back to CPU, which means one sample's focus map can be computed with **two different operators in different regions** — a discontinuity that depends on memory pressure rather than on the tissue.
The port cannot do this, because both backends now run the same operator.

### What this means for "exactly the same results"

This is the one place where the port **deliberately does not** reproduce dev HEAD, and it is a bug fix rather than drift:

|                                               | Reference `cd87a5d`                        | Port `882ed8a`                   |
| --------------------------------------------- | ------------------------------------------ | -------------------------------- |
| CPU operator                                  | `scipy.gaussian_laplace` (fused, analytic) | same                             |
| GPU operator                                  | `laplace(gaussian_filter(...))` shim       | `cupyx.gaussian_laplace` (fused) |
| Self-consistent across backends?              | **No** (~7 % on `lap_var_median_raw`)      | Yes (1 ULP)                      |
| OOM fallback can switch operators mid-sample? | **Yes**                                    | No                               |

So the honest answer to the requirement is:

- **On CPU the port is bit-for-bit identical to dev HEAD.** Proven twice, most strictly in the single combined image where neither arm was shadowed.
- **On GPU it is deliberately different, because dev HEAD's GPU path is wrong.** Matching it exactly would mean reintroducing an operator mismatch that grades GPU tiles systematically blurrier than CPU tiles and can switch operators mid-sample on memory pressure.

This is a decision for the user rather than something to silently resolve either way. The options are to fix the reference (adopt the fused operator upstream, which makes the two identical on both backends), or to accept the port as intentionally divergent-and-correct on GPU and record it as a known deviation. Reintroducing the shim in the port to force byte-equality would knowingly restore a defect and is not recommended.

## FINAL: the controlled GPU A/B, with both containers verified

Both arms ran in `0.0.16-equiv1`, and this time the container was **verified from the task record** rather than assumed:

| Arm                   | Workflow         | Container actually used                                | Machine     | Compute |
| --------------------- | ---------------- | ------------------------------------------------------ | ----------- | ------- |
| A reference `cd87a5d` | `1vzu3lIL2FIwrV` | `.../altoslabscom/xenium-processing-gpu:0.0.16-equiv1` | g6.16xlarge | 150 s   |
| B port `882ed8a`      | `1gRG4RHH9WlQp0` | `.../altoslabscom/xenium-processing-gpu:0.0.16-equiv1` | g6.16xlarge | 149 s   |

Comparable compute times on identical machine types confirm neither arm silently fell back to CPU, and both report `image_qc_status.json` = `{"status": "ok"}`.

### The environment is now genuinely refuted

The reference's GPU answer is **bit-identical across two verifiably different images**:

| Metric                            | Reference in ECR image | Reference in combined image |
| --------------------------------- | ---------------------- | --------------------------- |
| `lap_var_median_raw`              | 1926.1181640625        | 1926.1181640625             |
| `rois_blurred_gmm`                | 489100                 | 489100                      |
| `pct_blurred_gmm_tissue_filtered` | 35.03838550870625      | 35.03838550870625           |
| `usable_tissue_frac`              | 0.6495550075839401     | 0.6495550075839401          |

The earlier version of this claim was withdrawn because the container had not been verified and the run turned out to have used the ECR image both times. It now holds on evidence: two different images, container confirmed per task, identical output. **The cause is the reference's code, exactly as the operator analysis predicts.**

### The divergence is precisely scoped

Across the six shared metric JSONs:

|                                              | Count   |
| -------------------------------------------- | ------- |
| Keys identical                               | **554** |
| Differing, inside the Laplacian/blur cascade | 56      |
| Differing, outside it                        | **2**   |

And both of the two are `snr.components.SNR_roi_neg_spatial.moran_p_sim` (ref 0.26, port 0.21) — the permutation p-value documented above as irreproducible. The reference alone has now produced 0.34, 0.16, 0.22 and 0.26 for it across four runs, so it is noise on a null result and not a difference between implementations.

**Every genuine difference is downstream of the one operator substitution.** Nothing else in the QC output moved: all SNR components, all intensity statistics, the tissue masks, the ROI and cell totals, and all 40 figure geometries are identical.

The direction confirms the mechanism independently. The shim's response carries ~0.8x the variance, so the reference grades tissue systematically blurrier:

| Metric                     | Reference (shim) | Port (fused) |
| -------------------------- | ---------------- | ------------ |
| `cells_blurred_gmm_2d_roi` | 2284             | 2118         |
| `pct_blurred_gmm_2d_roi`   | 1.8089           | 1.6772       |
| `lap_var_median_raw`       | 1926.12          | 2066.91      |

Lower Laplacian variance, more cells called blurred — which is what a systematically under-responsive operator must produce, and it is what is observed.

## SUPERSEDED: "the code delta cannot explain the GPU divergence" — and why the audit missed it

> **This section's conclusion is wrong and is kept only to show how.** The code delta _does_ explain the divergence; see "ROOT CAUSE" above. The reasoning below was sound given its inputs, and the inputs were incomplete.

### Why a 102-of-114 clean symbol audit missed the operator

The symbol audit paired symbols **by name** and diffed the bodies with `difflib`. `cupy_gaussian_laplace` appears nowhere in its report, and the reason is structural rather than an oversight in reading it:

- In the reference it is a **`def`** at module scope, nested inside a `try:` block (`image_qc.py:81-85`).
- In the port it is an **`import ... as`** binding the same name (`qc.py:78-80`).

A pairing that collects function _definitions_ and matches them by name can never pair those two: the port defines no function called `cupy_gaussian_laplace`, so the reference's `def` had no counterpart and simply dropped out of the paired set. It was not reported as "differing" because it was never reported at all — and "nothing present in the reference was dropped from the port" was true of _functions_ while false of the _operator_, because the operator moved from a definition to an import.

**The methodology lesson, since "102 of 114 identical" was the whole basis for exonerating the code:** symbol-level pairing tests whether shared functions agree. It does not test whether the _set_ of names is the same, nor what a name is bound to when it is bound by an import rather than a definition. An equivalence audit needs to diff the module-scope import and assignment statements — including inside `try`/`except` — with at least as much care as the function bodies, because a one-line import swap can change every number in the output while leaving every paired function byte-identical. That is exactly what happened here.

### The original (incorrect) reasoning, preserved

Re-derived independently of the earlier audit, by AST-normalising both files (docstrings **and** all annotations stripped, `AnnAssign` folded to `Assign`, decorators dropped) and comparing all 114 shared symbols: **102 identical, 12 differing**. Of those 12:

| Symbol                                                                                         | Substantive change                                                          | Numeric effect                                                       |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| `BlockMeanAccumulator`, `CentrePixelSampler`, `LabeledSumAccumulator`, `RoiOtsuSnrAccumulator` | none — annotation-only (`'Cls'` -> `Cls`)                                   | **inert**                                                            |
| `load_roi_blur_threshold`                                                                      | `open(f, 'r')` -> `open(f)`                                                 | **inert**                                                            |
| `calculate_ccfs_measurements`                                                                  | `regionprops_table` import moved local                                      | **inert**                                                            |
| `generate_tissue_mask`                                                                         | `nsitk` import moved local                                                  | **inert**                                                            |
| `save_versions_file`                                                                           | `nsitk` import moved local                                                  | **inert**                                                            |
| `plot_grid_roi_focus_heatmap`                                                                  | two imports reordered                                                       | **inert**                                                            |
| `assess_raw_intensity_quality`                                                                 | `_yaml_keys` map -> `_yaml_channel_cfg`                                     | inert for the shipped YAML (verified: resolves to the same sub-dict) |
| `main`                                                                                         | split into `main` + `run_image_qc` (13 lines of substance)                  | inert                                                                |
| `_compute_channel_maps_tiled`                                                                  | **commit `9771f53`** — `with cp.cuda.Device(gpu_id)` and `del` moved inside | the only candidate                                                   |

Critically, the whole GPU tiling and Laplacian chain is **byte-identical**: `_compute_tile_grid`, `_compute_adaptive_strip_height`, `_compute_channel_maps_on_gpu`, `_process_tile_for_consumers` and `compute_laplacian_variance_map`. There is no difference in halo/overlap arithmetic and the four consumer accumulators differ only in type hints.

> **Correction.** Every clause above is still literally true, and the conclusion drawn from it was still wrong. Those functions _call_ `cupy_gaussian_laplace`, and that name resolves to a different operator in each codebase. Identical call sites invoking a differently-bound callee produce different numbers, which is precisely the case the audit could not see. "No difference in the Laplacian kernel" was the false step: the kernel is not in these functions, it is in what the name was bound to.

That leaves a contradiction: at `image_qc_gpus: 1`, `cp.cuda.Device(0)` is a no-op, so both sides execute effectively identical code — yet they reproducibly disagree on `lap_var_median_raw` (1926.1181640625 vs 2066.908203125). **No code path explains that.** Combined with the CPU runs being exactly identical in one shared image, the remaining variable is the environment.

### The unrecorded GPU stack

`versions.yml` is identical for both arms and records the host numeric stack:

```text
python 3.11.0 | numpy 2.2.6 | pandas 2.3.2 | scikit-image 0.25.2
tifffile 2025.9.20 | zarr 2.18.7 | napari-simpleitk-image-processing 0.4.9
```

but it does **not** record `cupy` or CUDA at all — and the GPU Laplacian is computed by `cupyx.scipy.ndimage`. The Docker Hub image carries:

```text
cupy-cuda12x              14.0.1
nvidia-cuda-runtime-cu12  12.9.79
```

A differing CuPy/CUDA between the two `0.0.16` images would produce exactly the observed signature: GPU-only, reproducible, device-count-independent, confined to GPU-computed quantities, with the CPU path unaffected.

**This is a reproducibility gap worth fixing on its own:** a QC pipeline whose numbers depend on the GPU stack should record `cupy` and the CUDA runtime in `versions.yml`. As it stands, two runs can differ numerically with byte-identical recorded versions.

## DECIDED: everything follows upstream

The user's instruction was "everything follow upstream", which settles the open `image_qc_snr_no_moran` question and turns it into an alignment task.

Auditing all ten `image_qc_*` pipeline parameters against `nf-xenium-processing` `origin/dev` found **exactly one** deviation:

| Parameter               | Upstream | spatialxe (before) | Now     |
| ----------------------- | -------- | ------------------ | ------- |
| `image_qc_snr_no_moran` | `false`  | `true`             | `false` |

The other nine were already identical, and `conf/roi_image_qc_thresholds.yaml` is **byte-identical** to upstream's (344 lines).
That last point retires a loose end: the `critical_threshold` 500-vs-50 and `pct_warn_threshold` 15-vs-40 gaps seen earlier were never a content deviation, only the difference between passing the thresholds file and not passing it.

Before flipping the default, `esda` was verified present in the image, because upstream's own schema description warns that Moran requires it: `esda` 2.9.0, `libpysal` 4.14.1, `pysal` 26.1. Turning Moran on by default therefore cannot fail for lack of the dependency.

Four files changed:

| File                                              | Change                                                                  |
| ------------------------------------------------- | ----------------------------------------------------------------------- |
| `nextflow.config:127`                             | `true` -> `false`                                                       |
| `nextflow_schema.json:480`                        | `"default": true` -> `false`, description matched to upstream's wording |
| `modules/local/image_qc/main.nf:83`               | comment said `true (default)`, which the flip made false                |
| `modules/local/image_qc/tests/nextflow.config:20` | `true` -> `false`                                                       |

The test-fixture change is worth singling out. That file's own header states its values "reproduce upstream nf-xenium-processing defaults", yet the line read `ext.snr_no_moran = true   // true = Moran's I off (upstream default)`.
Upstream's default is `false`, so the comment asserted the opposite of the truth and the fixture was pinning the non-default path — meaning the nf-test suite never exercised the default. Fixing it serves the file's stated intent as much as the instruction.

Three checks were run before accepting that flip, because turning Moran on could have introduced a flaky test rather than better coverage:

| Risk                                            | Check                                                  | Result                                                                                       |
| ----------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Missing dependency                              | `esda`/`libpysal` in `quay.io/dongzehe/image_qc:1.0.0` | present (2.9.0 / 4.14.1)                                                                     |
| Snapshot churn                                  | `modules/local/image_qc/tests/main.nf.test.snap`       | **no snapshot file exists**                                                                  |
| Snapshotting the non-reproducible `moran_p_sim` | assertions in `main.nf.test`                           | existence-only (`process.success`, `.exists()`, `isDirectory()`); no value or md5 assertions |

So the flip adds coverage of the default path without putting `moran_p_sim` under any assertion.

**Consequence to expect:** with Moran on by default, `snr_metrics.json` and `roi_qc_metrics.json` gain `moran_i`, `moran_p_sim`, `moran_verdict`, `method_secondary` and the `moran_subsample` block, and lose `moran_note`. Any nf-test snapshot covering those files will need regenerating, and `moran_p_sim` is the key documented above as not reproducible, so it should not be snapshotted on value.

**Not changed:** the `feat/spatialqc-package` worktree carries the identical deviation at its own `nextflow.config:127`, `nextflow_schema.json` and `modules/local/image_qc/tests/nextflow.config:20`. It is a different branch and it is the code currently under test in the A/B above, so it was left alone rather than edited mid-comparison. It needs the same four-line change when that branch is next touched.

## RESOLVED: the fix is written, committed and being validated

The operator decision was made — follow upstream by _fixing_ upstream, rather than by reintroducing the shim downstream.

### Where the operator lived, in all three codebases

The port was not built on a stale commit; this was checked directly:

| Codebase                                                          | GPU operator                                    |
| ----------------------------------------------------------------- | ----------------------------------------------- |
| `nf-xenium-processing` dev HEAD `cd87a5d`                         | `def cupy_gaussian_laplace(...)` — the shim     |
| spatialxe `bin/image_qc.py` (`eb56727`, re-synced from `cd87a5d`) | the shim, **byte-identical to upstream**        |
| spatialqc package `882ed8a`                                       | `from cupyx... import gaussian_laplace` — fused |

So the port really is a faithful copy of dev HEAD everywhere except this one line, and spatialxe's own re-synced script still carries upstream's shim verbatim. The deviation was introduced deliberately in the `spatialqc` packaging branch, with a rationale recorded in its own comment. That also explains the otherwise puzzling shape of the result — the substituted name is referenced **only inside the `if use_gpu:` branch**, so a GPU-only one-line change gives exactly what was measured: 0 differing values on CPU, 56 on GPU.

### The commits

| Repo / branch                                 | Commit    | Change                                                           |
| --------------------------------------------- | --------- | ---------------------------------------------------------------- |
| `nf-xenium-processing` `fix/gpu-log-operator` | `adc33f4` | replace the GPU shim with `cupyx.scipy.ndimage.gaussian_laplace` |
| spatialxe `feat/qc-from-upstream-dev`         | `5bea5d4` | `image_qc_snr_no_moran` -> upstream's `false` (4 files)          |
| spatialxe `feat/spatialqc-package`            | `db0fce7` | same alignment on the package branch                             |

`adc33f4` is pushed to `origin/fix/gpu-log-operator`; a PR can be opened at
`https://github.com/altos-labs/nf-xenium-processing/pull/new/fix/gpu-log-operator`.
`dev` was not touched.

The patch deletes five lines and adds one import. Both call sites (`:1087`, `:1229`) pass `sigma=` and need no change; `cupy_laplace` stays imported for the `lap_sigma <= 0` branch, which was always consistent across backends.

### Validation in flight

Workflow `du9CFGsdE61co` runs the **patched** reference at `adc33f4` in the same `0.0.16-equiv1` image, at 1 GPU, with the same 180 GB / 30 cpu and the same samplesheet as the verified arm A. The prediction on record: its GPU output should now match the port's GPU output, leaving only `moran_p_sim`.

### Resource control, for the record

All GPU arms were allocated identically, confirmed from the task records rather than from the submitted config:

| Run                    | Requested       | Peak RSS | Attempt |
| ---------------------- | --------------- | -------- | ------- |
| A ref, combined image  | 180 GB / 30 cpu | 30.09 GB | 1       |
| B port, combined image | 180 GB / 30 cpu | 30.04 GB | 1       |
| A ref, ECR image       | 180 GB / 30 cpu | 30.74 GB | 1       |

All `attempt=1`, so the retry doubling never engaged and memory was literally identical. Peak RSS ~30 GB against a 180 GB request is 17 % utilisation, which means **the GPU-OOM fallback never fired in any run** — so the 56 differences are the operator substitution acting uniformly, not a partial mid-sample fallback. (It also means the 180 GB request is over-provisioned by ~6x on a 256 GB node, worth trimming for scheduling reasons unrelated to correctness.)

## VALIDATED: with `adc33f4` the two implementations are identical on GPU too

Workflow `du9CFGsdE61co` ran the patched reference at `adc33f4` in `0.0.16-equiv1` (container verified from the task record), 1 GPU, 180 GB / 30 cpu, 152 s — the same configuration as the verified arm A, changing only the operator.

### The patch moves the reference exactly onto the port's answer

| Metric                            | Reference before   | Reference after        | Port               |
| --------------------------------- | ------------------ | ---------------------- | ------------------ |
| `lap_var_median_raw`              | 1926.1181640625    | **2066.908203125**     | 2066.908203125     |
| `rois_blurred_gmm`                | 489100             | **487910**             | 487910             |
| `pct_blurred_gmm_tissue_filtered` | 35.03838550870625  | **34.69194064438745**  | 34.69194064438745  |
| `usable_tissue_frac`              | 0.6495550075839401 | **0.6530136336243664** | 0.6530136336243664 |

Not close — equal to every printed digit. And the "after" column is also the CPU answer, to within one float32 ULP, so the patch reconciles the reference with _itself_ across backends at the same time.

### Before and after, on the same tool

|                                                    | Before `adc33f4`    | After `adc33f4`   |
| -------------------------------------------------- | ------------------- | ----------------- |
| Integer keys moved                                 | **22**              | **0**             |
| Float keys differing                               | 34                  | **0**             |
| Max relative delta                                 | 7.688e-01           | —                 |
| `SNR_roi_tx.parquet` `dapi_lap_var` rows differing | **602504 / 712236** | **0 / 712236**    |
| `blur_prob_gmm_2d` rows differing                  | 311581 / 712236     | **0 / 712236**    |
| `is_blurred_gmm_2d` rows differing                 | 1206 / 712236       | **0 / 712236**    |
| Tables matching                                    | 2 of 5              | **5 of 5**        |
| Verdict                                            | not equivalent      | **bit-identical** |

The 712,236-row Laplacian column that was the origin of the whole cascade now agrees on every single row.

### What is left

Exactly two keys differ in the entire comparison, and they are the same key in two files:

```text
roi_qc_metrics.json : snr.components.SNR_roi_neg_spatial.moran_p_sim  ref=0.27  port=0.21
snr_metrics.json    : components.SNR_roi_neg_spatial.moran_p_sim      ref=0.27  port=0.21
```

`moran_p_sim` is the permutation p-value that is irreproducible by construction — the reference alone has now produced 0.34, 0.16, 0.22, 0.26 and 0.27 for it across five runs with `seed: 42` reported every time, on a null result (`moran_i` ~ -2e-04). It is sampling noise, not a difference between implementations, and recommendation 3 covers making it reproducible.

### The requirement, answered

**With `adc33f4` applied, spatialqc generates the same results as `nf-xenium-processing` on both CPU and GPU** — bit-identical on every metric, table and figure geometry, with the single exception of a Monte-Carlo p-value that no two runs of either implementation reproduce.

Without the patch, the answer is: identical on CPU, and different on GPU in 56 metrics all traceable to one line, where the port was right and dev HEAD was wrong.

## ANSWERED: zero differences, on both backends

With the two fixes applied to both codebases, `nf-xenium-processing` and `spatialqc` produce **identical output on GPU**, with nothing excused or set aside.

Arms verified from the task records, not from the submitted config:

| Arm         | Revision                    | Container actually used                  | Resources              |
| ----------- | --------------------------- | ---------------------------------------- | ---------------------- |
| A reference | `d084145`                   | `...xenium-processing-gpu:0.0.16-equiv2` | 180 GB / 30 cpu, 155 s |
| B port      | wheel `627e078` in image v2 | `...xenium-processing-gpu:0.0.16-equiv2` | 180 GB / 30 cpu, 161 s |

The port's code ships in the wheel rather than in git, so its fix arrived via image `0.0.16-equiv2` (digest `sha256:3fe2f66a12474e0c34032a2e5aad3844e5a67e1fc737681b16c23b385c741fdb`) while the pipeline revision stayed `882ed8a`. Before launching, the image was checked for the numeric stack being byte-identical to base and for both fixes being present in the _installed_ wheel, not merely in the source tree.

### Result

| Check                                          | Result                                                |
| ---------------------------------------------- | ----------------------------------------------------- |
| File sets                                      | identical, 55 files each                              |
| JSON numeric leaves                            | **no differences in any shared JSON**                 |
| Integer keys (counts, totals, classifications) | 236 compared, **0 moved**                             |
| Float keys                                     | 242 exact, **0 differing**                            |
| Tables                                         | **all 5 match exactly**                               |
| `SNR_roi_tx.parquet`                           | 712,236 rows x 33 cols — **0 of 23.5 M cells differ** |
| Figures                                        | 40 shared, 0 differing in dimensions                  |
| Verdict                                        | **bit-identical — not one compared value differs**    |

The key that was the last survivor is now equal, and equal to a _reproducible_ value rather than to noise:

```text
components.SNR_roi_neg_spatial.moran_i        ref = port = -0.00021797113894864514
components.SNR_roi_neg_spatial.moran_p_sim    ref = port = 0.17
components.SNR_roi_neg_spatial.moran_verdict  ref = port = 'PASS'
```

### The two defects, and what each cost

| #   | Defect                                                                                                        | Symptom                                                                                                                                            | Fix                                                         |
| --- | ------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 1   | GPU used `laplace(gaussian_filter(x, σ))`; CPU used `scipy.gaussian_laplace`                                  | 56 metrics differed; 602,504 / 712,236 ROIs differed on `dapi_lap_var`; 1,206 ROIs reclassified; the two backends of _one_ pipeline disagreed ~7 % | `cupyx.scipy.ndimage.gaussian_laplace`                      |
| 2   | `esda.Moran` permutations drew from numpy's unseeded legacy global RNG while the output reported `"seed": 42` | `moran_p_sim` changed every run: 0.34, 0.16, 0.22, 0.26, 0.27 on unchanged input                                                                   | seed the legacy stream around the call, restore state after |

Both were real defects in the reference, both are fixed in both codebases, and neither was worked around in the comparison tooling.

### Final commit set

| Repo / branch                                 | Commit    | Change                                      |
| --------------------------------------------- | --------- | ------------------------------------------- |
| `nf-xenium-processing` `fix/gpu-log-operator` | `adc33f4` | fused GPU Laplacian-of-Gaussian             |
| `nf-xenium-processing` `fix/gpu-log-operator` | `d084145` | seed the Moran permutation test             |
| spatialxe `feat/qc-from-upstream-dev`         | `5bea5d4` | `image_qc_snr_no_moran` -> upstream `false` |
| spatialxe `feat/spatialqc-package`            | `db0fce7` | same alignment                              |
| spatialxe `feat/spatialqc-package`            | `627e078` | seed the Moran permutation test             |

Both `nf-xenium-processing` commits are pushed to `origin/fix/gpu-log-operator`; `dev` was never touched. PR: `https://github.com/altos-labs/nf-xenium-processing/pull/new/fix/gpu-log-operator`.

### Migration note

GPU-derived QC output changes with `adc33f4`, and `moran_p_sim` changes with `d084145`. On the reference sample: `lap_var_median_raw` 1926.12 -> 2066.91 and ~1,190 fewer ROIs are called blurred, so samples graded near a blur threshold can change verdict. CPU results are unaffected by either fix. Results from before and after these commits are not directly comparable and the change deserves a CHANGELOG entry and a version bump.

## Tower launch traps — diagnosed and fixed

The first GPU A/B attempt (`pmZm9OvOTrFTF`, `3Uc5VRLlayQCE2`) failed on both arms, for two _different_ queue faults. Both are now fixed or worked around, and the root causes are worth keeping.

### 1. The API error body was silently discarded (fixed)

`scripts/tower-launch.sh` looked like it printed the failure body:

```bash
echo "Error: POST $url returned HTTP $http_code" >&2
echo "$body" | jq . 2>/dev/null || echo "$body" >&2
```

`jq .` writes to **stdout**, and the caller runs `response=$(tower_post ...)`, so the body was captured into the variable rather than shown. `exit 1` inside a command substitution only exits the subshell, and `set -euo pipefail` then aborted the script on the failed assignment — discarding the captured body. The `|| echo ... >&2` fallback never ran, because `jq` succeeds on valid JSON. Net effect: a launch rejection surfaced as a bare `HTTP 400` with the reason thrown away.

Fixed in both `tower_get` and `tower_post` by grouping the body to stderr:

```bash
{ echo "$body" | jq . 2>/dev/null || echo "$body"; } >&2
```

This is what turned an opaque `HTTP 400` into `The selected compute environment (deleted-33747942085129513 - 5ixknRlHbJM1T0Fr2iRPW7) is not available any more`.

### 2. Stale compute-environment list, no preflight check (fixed)

The usage text advertised `5ixknRlHbJM1T0Fr2iRPW7` as "GPU big" and an example recommended it for Cellpose — but it is **deleted**. A deleted environment is still readable at `/compute-envs/<id>`, so the queue-name lookup succeeded and the run only failed at `POST /workflow/launch`.

Fixed by making the platform the source of truth rather than a comment that rots:

- `list_compute_envs()` and a `--list-compute-envs` flag print the live workspace listing.
- `validate_compute_env()` runs before any payload is built, and on a missing environment exits with the available alternatives. Verified: launching with the dead ID now fails fast and does **not** submit.
- The dead ID was removed from the usage text and examples.

### 3. `-c` did not govern GPU tasks — the real cause of arm B's failure (fixed)

Arm B failed with `Missing AWS Batch job queue -- provide it by using the process 'queue' directive`, despite `-c` naming a valid environment. The `aws` profile resolves GPU queues from **params**, not from the process scope:

```groovy
withLabel:process_gpu_qc { queue = { params.gpu_queue ?: null } }
```

A `withLabel` selector outranks the generic `process { queue = ... }` block the script emits, and `gpu_queue` defaults to `null` (`nextflow.config:33`) — so every GPU task got no queue at all. This is a parameterised design, and better than hardcoding, but the launch script never supplied the parameter.

Fixed with `inject_queue_params()`, which derives `gpu_queue` and `cellpose_queue` from the chosen compute environment unless the params file already sets them. Verified in a dry run:

```json
"gpu_queue": "TowerForge-4WpM1mBmA2qSC6CxukQ1yx",
"cellpose_queue": "TowerForge-4WpM1mBmA2qSC6CxukQ1yx"
```

### 4. The reference pins a deleted queue — reported, deliberately NOT fixed

Arm A failed with `JobQueue TowerForge-5ixknRlHbJM1T0Fr2iRPW7 not found`, because `nf-xenium-processing` hardcodes the deleted environment at **`conf/base.config:128`** (and at `:68`, `:142`, `:234` for other labels), inside `withLabel:process_gpu_qc` — which outranks the launch script's global `process.queue`.

This was **not** fixed in that repo, on purpose: it is the baseline spatialqc must reproduce, and editing it mid-verification would change the very thing under comparison. Instead arm A was relaunched by direct API POST with a `configText` that restates the reference's own `process_gpu_qc` values verbatim and changes only the queue:

```groovy
withLabel:process_gpu_qc {
  cpus   = 30
  memory = { 180.GB * (Math.pow(2, task.attempt - 1) as int) }
  time   = { 4.h * (Math.pow(2, task.attempt - 1) as int) }
  queue  = 'TowerForge-4WpM1mBmA2qSC6CxukQ1yx'   // was the deleted env
  accelerator = { params.use_gpu ? (params.image_qc_gpus as int) : null }
}
```

Restating the whole block rather than setting `queue` alone is deliberate: `withLabel` merge-vs-replace semantics could otherwise drop `accelerator`, which is what actually allocates the GPUs — a silent downgrade to a CPU run. Queue and resource directives do not affect numerics, so the reference's results are unchanged.

**Recommended follow-up for `nf-xenium-processing` owners:** remove the four hardcoded `TowerForge-5ixknRlHbJM1T0Fr2iRPW7` queues from `conf/base.config` and take the queue from the compute environment (or a param, as spatialxe does). Any GPU run of that pipeline on the default config currently fails.

### Relaunched A/B

| Arm           | Pipeline @ commit                  | Workflow ID      | Queue source                           |
| ------------- | ---------------------------------- | ---------------- | -------------------------------------- |
| A — reference | `nf-xenium-processing` @ `cd87a5d` | `2Mce5R6pieSt8B` | `configText` override (repo untouched) |
| B — port      | `spatialxe` @ `882ed8a`            | `2ewDL90CwV58z9` | `gpu_queue` param from `-c`            |

## Status

- [x] Reference version identified and blob-verified (`origin/dev` = `cd87a5d`)
- [x] Determinism confirmed (`random_state=0`, `default_rng(42)`)
- [x] Vendored `xenium_helpers` surface audited — equivalent
- [x] Comparison script written, self-tested, and its own suffix-match bug fixed
- [~] Code-level symbol audit — every _paired_ difference inert except `9771f53`; **the audit was incomplete**, it could not see `cupy_gaussian_laplace` changing from a `def` to an `import`, which is the real cause
- [x] **CPU path proven identical on a real bundle — 505 numeric JSON leaves, 5 tables, 54 files**
- [x] GPU A/B on Tower at 4 GPUs (g6.24xlarge) and at 1 GPU
- [x] Multi-GPU caveat resolved: `9771f53` is inert here — each side is GPU-count-stable on every substantive metric
- [x] Port's own CPU-vs-GPU outputs measured, not assumed: **236/236 integer keys exact**, max float delta 3.68e-06, no classification threshold crossed
- [x] Threshold _inputs_ identified as a local-vs-Tower config delta; both Tower arms confirmed to share thresholds
- [x] `moran_p_sim` found non-reproducible in the reference itself (0.34 @ 4 GPU vs 0.16 @ 1 GPU, seed 42 both) — excluded from equality expectations
- [x] Divergence localised to the single reference-on-GPU cell; spatialqc exonerated
- [x] **Combined image built and verified:** `0.0.16-equiv1` runs both scripts; numeric stack byte-identical to base
- [x] **Single-image CPU A/B: bit-identical** — 0 differing values across 39 files, 5 tables, 25 figures
- [x] `image_qc_snr_no_moran` decided: everything follows upstream, all 10 params now identical
- [x] Two harness bugs fixed (read-only cwd, bundle-hardcoded sanity) plus a NaN bug in `residual_deltas.py`
- [x] Image pushed to Docker Hub; both GPU arms run, containers verified from the task record
- [x] **Environment refuted on evidence:** reference GPU output bit-identical across two different images
- [x] **Root cause found:** the reference applies `laplace(gaussian_filter(...))` on GPU vs `scipy.gaussian_laplace` on CPU — different operators, ~0.8x response variance
- [x] Divergence scoped: 554 keys identical, 56 differing all downstream of that operator, 2 are Monte-Carlo `moran_p_sim` noise
- [x] **Decision made, implemented and VALIDATED:** operator fixed on `fix/gpu-log-operator` (`adc33f4`, pushed); run `du9CFGsdE61co` proves ref == port on GPU, 0 integer keys moved, 0 floats differing, all 5 tables exact
- [x] `dapi_lap_var` agreement went from 109732/712236 rows to **712236/712236**
- [x] **Second defect found and fixed:** `esda.Moran` permutations came from numpy's unseeded legacy global RNG despite the output reporting `seed: 42`
- [x] **FINAL: bit-identical on GPU** — 0 differing values, all 5 tables exact, 0 of 23.5 M parquet cells differ (`2f63W19rCzVwEb` vs `3nCyF41Cjj1WFP`)
- [x] Moran alignment committed on both spatialxe branches (`5bea5d4`, `db0fce7`); `make check` passes

## Recommended follow-ups

Ordered by value, with the reason each matters:

1. ~~**Build one image carrying both the spatialqc wheel and dev-HEAD's `xenium_helpers`.**~~ **DONE** — `altoslabscom/xenium-processing-gpu:0.0.16-equiv1`, built and verified above. What remains is pushing it to Docker Hub so Tower can pull it; that push was blocked by a permission prompt and is the one outstanding action.
   The image is also the only way to attribute the reference's cross-backend shift.
   That shift is real and large — `lap_var_median_raw` 1926.12 on GPU vs 2066.91 on CPU, ~7 % — and it propagates into the blur classification, but reference-on-GPU and reference-on-CPU ran in _different images_, so the shift cannot presently be pinned on either the code or the environment.
   **Update: this is now resolved and the defect is identified.** The image was refuted as the cause and the reference's GPU Laplacian operator is the cause; see "ROOT CAUSE". This item is complete, and the remaining question is the operator decision below, not attribution.
2. ~~**Decide the `image_qc_snr_no_moran` default.**~~ **DONE** — everything follows upstream; all ten parameters now match `origin/dev`. Regenerate any nf-test snapshot covering `snr_metrics.json` or `roi_qc_metrics.json`, and do not snapshot `moran_p_sim` on value.
3. **Make the Moran permutation test reproducible.**
   `moran_p_sim` is 0.34 at 4 GPUs and 0.16 at 1 GPU in the reference with `seed: 42` reported both times, so the recorded seed does not currently reproduce the result. The cause is not isolated — parallel decomposition and an unseeded global RNG both fit — so the first step is two same-configuration runs to tell them apart.
   This affects the reference and the port equally.
4. **Record `cupy` and the CUDA runtime in `versions.yml`.**
   Today two runs can differ numerically while reporting byte-identical versions, because the GPU stack is not captured and the Laplacian runs through `cupyx`.
5. **Pin containers by digest, not tag.**
   `docker.io/altoslabscom/xenium-processing-gpu:0.0.16` and ECR `nextflow/containers:xenium-processing-gpu-0.0.16` are different software under one version string.
6. **Remove the dead queues from `nf-xenium-processing/conf/base.config`** (`:68`, `:128`, `:142`, `:234`).
   Any GPU run of that pipeline on its default config currently fails.
7. **Treat `image_qc_status.json` as the QC gate in any comparison tooling.**
   A workflow reports SUCCEEDED even when the analysis exited 1 and wrote no metrics.
8. **Exclude threshold _inputs_ from equivalence tooling, and compare them separately.**
   `critical_threshold` and `pct_warn_threshold` are configuration read at runtime; when they differ, the 23 counts and statuses derived from them differ too and swamp the real signal.

## Artifacts and housekeeping

Run outputs are left in place rather than cleaned up, so this can be re-checked or reused.

| S3 prefix under the spatialxe scratch bucket | Workflow ID      | What it is                                                                      |
| -------------------------------------------- | ---------------- | ------------------------------------------------------------------------------- |
| `equiv_ab_ref_devhead`                       | `2Mce5R6pieSt8B` | reference @ `cd87a5d`, 4 GPUs                                                   |
| `equiv_ab_port_spatialqc`                    | `2ewDL90CwV58z9` | port @ `882ed8a`, 4 GPUs                                                        |
| `equiv_1gpu_ref`                             | `1dvzIdir8LJ1Sz` | reference, 1 GPU                                                                |
| `equiv_1gpu_port`                            | `4GMMVr85SABJV6` | port, 1 GPU                                                                     |
| `equiv_1gpu_ref_sameimage`                   | `3NFtjYuS4FgqOr` | reference in the Docker Hub image — SUCCEEDED but analysis exited 1, no metrics |
| `equiv_1gpu_port_ecrimage`                   | `2uBAuvJmm8udxy` | port in the ECR image — FAILED, no spatialqc wheel                              |
| `equiv_1gpu_port_ecr2`                       | `zpL2JTqQkg2jw`  | second attempt at the above — FAILED                                            |

Two earlier launches, `pmZm9OvOTrFTF` and `3Uc5VRLlayQCE2`, failed on the queue traps described above and produced no outputs.

Also note, for whoever picks this up:

- `scripts/tower-launch.sh` is **untracked** and the four fixes in it are **uncommitted**.
  They are real fixes and they are what made the A/B launches work, but committing internal tooling is the user's call, not mine.
- This file and `2026-09-10_REVIEW_spatialqc-symbol-diff-vs-dev-head.md` are in `docs/plans/`, which the project's nf-core PR guidelines exclude from upstream PRs.
- `nf-xenium-processing` was deliberately left **unmodified** throughout, because it is the comparison baseline. The deleted-queue override was applied at launch time via `configText`, not by editing its config.

## Reproducing

The tooling is preserved in `scripts/qc-equivalence/` (untracked, like the rest of `scripts/`):

| File                                    | Purpose                                                                                                   |
| --------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `compare_outputs.py`                    | exact-equivalence verdict over two output trees — JSONs, tables, figure dimensions                        |
| `residual_deltas.py`                    | when the verdict is "not identical", quantifies by how much and separates config inputs from measurements |
| `in_container_ab.sh`                    | runs reference and port back-to-back in one container, so the two cannot differ by environment            |
| `in_container_ref.sh`                   | runs the reference alone with dev-HEAD's own `xenium_helpers` shadowing the container's stale copy        |
| `*_compare.txt`, `selfcheck_report.txt` | the reports the verdicts in this document are based on                                                    |

A note on why the two tools report different leaf counts for the same data.
`compare_outputs.py` counts every numeric leaf it compares (505 on the CPU pair) and skips nothing.
`residual_deltas.py` reports 236 integer plus 242 float keys on that same pair, because it classifies the threshold inputs and Moran keys out and ignores non-numeric leaves.
Both tools return "identical" on the CPU pair; the counts differ only in convention, not in coverage, and the unclassified tool is the one the equivalence verdict rests on.

The port's cross-backend numbers quoted above come from:

```bash
python3 scripts/qc-equivalence/residual_deltas.py <port-cpu-outdir> <port-gpu-outdir>
# 236 integer keys compared, INTEGER KEYS MOVED: 0, max relative delta 3.676e-06
```

```bash
SP=<scratchpad>
# reference, dev HEAD, with dev HEAD's own helpers
docker run -d --name refrun --pid=host --memory=120g \
  -v /home/dhe/qc_e2e/bundle:/data/bundle:ro \
  -v $SP/ref:/ref:ro -v $SP/helpers_root:/helpers:ro \
  -v $SP/in_container_ref.sh:/run_ref.sh:ro \
  -e OMP_NUM_THREADS=16 -e MKL_NUM_THREADS=16 -e OPENBLAS_NUM_THREADS=16 -e NUMBA_NUM_THREADS=16 \
  -e MPLCONFIGDIR=/tmp/mpl -e NUMBA_CACHE_DIR=/tmp/numba -e PYTHONUNBUFFERED=1 \
  docker.io/altoslabscom/xenium-processing-gpu:0.0.16 bash /run_ref.sh

# extract BEFORE removing anything, and confirm the files are on disk first
docker cp refrun:/tmp/ab/ref  $SP/out/ref
docker cp ab1:/tmp/ab/port    $SP/out/port

# then compare
python3 $SP/compare_outputs.py $SP/out/ref $SP/out/port
```

The comparison is exact-equality with two deliberate refinements:

- `NaN == NaN` counts as equal, because on both sides it means "not computed".
- A key whose _name_ looks volatile but whose _value_ is numeric and differs is reported as **SUSPECT and counted as a failure**, not absolved. Substrings like `date`, `dir` and `time` also match real metric names (`update_fraction`, `direction`), and a genuine numeric change hidden in a "volatile" bucket would otherwise read as a clean run. Only non-numeric volatile differences are treated as benign, and they are printed to be read rather than suppressed.

It also runs **positive** sanity assertions before diffing, because "no difference" between two identically-empty trees would be a worthless verdict. It checks `total_rois == 712236`, `rois_blurred_gmm == 487910`, `total_rois_tissue_filtered == 343489`, `total_cells == 126045`, that `image_qc_metrics.json` carries 40 top-level keys, and that `snr_metrics.json` has an `overall_snr_verdict` — on **both** sides.

Supporting evidence: the full symbol-by-symbol audit is preserved alongside this file as `2026-09-10_REVIEW_spatialqc-symbol-diff-vs-dev-head.md`.

Gotchas that cost time here, worth keeping:

- Background processes started with `nohup ... &` or `setsid` inside a tool call are killed when the call returns; `docker run -d` survives because the daemon owns the process.
- Overlapping `docker rm -f` / `docker kill` on the same container deadlocks that container's daemon lock while leaving the daemon itself responsive (`docker version` and `docker info` still return instantly).
- Container processes run as uid 1000, so they cannot be signalled from the host with `kill` as an ordinary user.
- `py-spy --native` on this workload falls ~20 s behind at 50 Hz and distorts both the profile and the run; ptrace attach to an already-running process is blocked (`ptrace_scope=1`, no `CAP_SYS_PTRACE`), though py-spy can still profile a process it spawns itself.
