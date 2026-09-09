# spatialqc: extraction to a standalone repository, and follow-ups

Status: the package exists and is wired into the pipeline inside this repository.
It is not yet published, and the QC containers do not yet contain it.
This document covers the remaining steps and the decisions that need a human.

## 1. RESOLVED: the GPU focus-score change needs no recalibration

> **Closed 2026-09-09 on real samples.**
> Two bundles from real Tower runs (`tower_launch/*.csv`), 1536 tiles each, compared fused versus shim through the real `fit_focus_gmm_2d` + `classify_roi_blur_2d`:
>
> | Sample          | Platform  | Blurry (fused) | Blurry (shim) | Delta        | Verdict     |
> | --------------- | --------- | -------------- | ------------- | ------------ | ----------- |
> | `v1_R2_control` | Xenium v1 | 0.1439         | 0.1582        | **-1.43 pp** | PASS → PASS |
> | `atera_breast`  | Xenium v2 | 0.2617         | 0.2591        | **+0.26 pp** | PASS → PASS |
>
> **Neither changes verdict, and both sit far below `focus_warn: 0.40`. Ship the fused operator as-is; do not recalibrate.**
>
> The harness reproduces a number the threshold YAML already records — `atera_breast` at 25.9-26.2% blurry against the YAML's *"observed tissue-filtered % blurry floor: ~27% even on best samples"* — which is a cross-check on the method, not only the result.
>
> The remaining gap is narrow: both samples pass with a wide margin, so neither exercises a sample sitting near 0.40, which is the only place a 1-2 pp shift decides anything. `tests/manual/real_calibration.py` runs the comparison if a borderline sample turns up.
>
> Full numbers and the two bundle-reading traps: [`2026-09-09_SPIKE_spatialqc-gpu-validation.md`](2026-09-09_SPIKE_spatialqc-gpu-validation.md) sections 4 and 5.

> **Update 2026-09-09, after GPU validation.**
> This section was written without GPU access — the machine's `/dev/nvidia*` nodes appeared two minutes after that session ended, so no CuPy branch had ever executed.
> The paths have now been run on 4x NVIDIA L4 / CuPy 14.0.1.
> Full numbers: [`2026-09-09_SPIKE_spatialqc-gpu-validation.md`](2026-09-09_SPIKE_spatialqc-gpu-validation.md).
>
> What changed in this section's conclusions:
>
> - **CPU and GPU now agree to correlation 1.00000000** (max relative deviation 0.0000%). The mixed-scale hazard from the GPU-OOM fallback is closed, confirmed empirically.
> - **`cupyx.scipy.ndimage.gaussian_laplace` confirmed present** on hardware, so the shim's justifying comment was indeed false.
> - **The divergence is content-dependent**: ~7% median on nuclei-like (DAPI-like) content versus 25.2% on hard synthetic edges. The table below is a third tile; the three are not corrections of each other.
> - **The shift is systematically larger on sharp tiles** — `spearman(blur_sigma, log_shift)` = -0.756, -0.740, -0.725 across three seeds. The fused operator therefore *widens* blur/focus separation, which is the safe direction and an argument for the fix on its own merits.
> - **`lap_focus_corr_warn`/`_fail` and `focus_median_warn` need no attention.** The first moves by at most +0.01; the second gates `focus_score` = `std**2/mean` of raw intensity (`image/qc.py:4608`), which never touches the Laplacian.
> - **The recommendation below still stands, but the reason has changed.** There is no systematic recalibration factor: the blurry-fraction delta across three seeds was -14.58, +2.08 and -0.69 pp — inconsistent in sign. The real risk is GMM *instability* on samples whose bimodality is marginal, where a ~15 pp swing is possible and `focus_warn` is 0.40. Also check `pct_blurred_gmm_2d_roi_warn: 20.0`, which is downstream of the same fraction.

This is the only item that changes numbers, and it should be signed off before release.

Image QC computed the Laplacian-of-Gaussian differently on the two backends.

| Backend | Operator                                   | Source                           |
| ------- | ------------------------------------------ | -------------------------------- |
| CPU     | `scipy.ndimage.gaussian_laplace(x, sigma)` | fused Gaussian second derivative |
| GPU     | `laplace(gaussian_filter(x, sigma))`       | hand-written shim, now removed   |

The shim was introduced with a comment stating that `cupyx` exposes no fused
`gaussian_laplace`.
It does: `cupyx.scipy.ndimage.gaussian_laplace`, documented as "Multi-dimensional Laplace filter using Gaussian second derivatives".

Measured divergence between the two formulations, on a synthetic tile with edges plus noise:

| `lap_sigma` | max relative deviation | correlation | `var(shim) / var(fused)` |
| ----------- | ---------------------- | ----------- | ------------------------ |
| 0.5         | 67%                    | 0.580       | 0.11                     |
| **1.0**     | **16.6%**              | **0.9937**  | **0.77**                 |
| 2.0         | 4.5%                   | 0.9997      | 0.94                     |

`lap_sigma` defaults to 1.0 in `data/image_qc_thresholds.yaml` and at every call site, so the middle row is the operative one.
The focus score is a variance-like reduction of this response, so GPU tiles were graded systematically differently from CPU tiles.
The shim is also on the production streaming path, and the GPU-OOM fallback switches to the CPU path mid-run, so a single sample could be graded on two scales.

Both backends now use the fused operator.

What this means for existing thresholds:

- Per-tile blur classification is **relative** to the sample: the primary path is a Gaussian-mixture posterior (`blur_prob_threshold: 0.5`) and the fallback is a percentile of that sample's own scores (`roi_focus_score_percentile: 5.0`). With correlation 0.9937 the within-sample re-ranking is small.
- The **sample-level** verdicts are absolute: the blurry-tile percentage cutoffs in the `focus` section of the threshold YAML were calibrated empirically on real samples. Those percentages will move, and the module runs under `label 'process_gpu_qc'`, so the historical calibration was most likely done against shim numbers.

Recommended: re-run image QC on two or three calibration samples and compare the reported blurry-tile percentages against the YAML cutoffs before release. **Done — see the RESOLVED block at the top of this section; no recalibration is needed.**

## 2. Container rebuild (blocking for docker-profile tests)

The module `container` directives point at tags that do not exist yet:

- `quay.io/dongzehe/image_qc:2.0.0`
- `quay.io/dongzehe/transcript_qc:2.0.0`

Both `environment.yml` files now pin `spatialqc==0.1.0` from PyPI, which is also not published yet.

Until both are done, `-profile docker` nf-tests for these two modules fail; `-stub` runs are unaffected.

Sequence:

1. Publish `spatialqc` 0.1.0 (see section 3), or build the containers from a local checkout with `pip install -e packages/spatialqc`.
2. Build and push the two images from the module `environment.yml` files.
3. Re-run the module nf-tests with `--profile=+docker`.

## 3. Extracting to a standalone repository

> **Done 2026-09-09.** The package now lives at <https://github.com/altos-labs/spatialqc> (internal), default branch `main`, six commits.
>
> The split was re-run from the post-fix HEAD, so `main` carries the device-scoping and `max_gpus` fixes and the GPU validation scripts — the pre-existing `spatialqc-standalone` branch predates all of them and is **superseded; do not push it**.
>
> Verified after pushing, not assumed:
>
> | Check                                    | Result                                                        |
> | ---------------------------------------- | ------------------------------------------------------------- |
> | Fresh `git clone`                        | `pyproject.toml`, `src/`, `tests/`, `.github/` at root         |
> | `python -m build --wheel` from the clone | `spatialqc-0.1.0-py3-none-any.whl`                            |
> | Package data in the wheel                | both threshold YAMLs (20952 / 17981 bytes) and `py.typed`      |
> | Test suite from the clone                | 247 passed, 1 skipped                                          |
> | Stray artefacts                          | no `build/`, `egg-info`, `__pycache__` or `.ruff_cache`        |
>
> The `.gitignore` `data/` trap noted in section 4.1 did not bite: both YAMLs travel in the wheel from a clean checkout.
>
> The repo's **first CI run** caught five mypy errors in `image/snr.py` that had never been type-checked before — fixed in `9fc1982`, and pre-existing rather than introduced by the port. Everything else passed on the first run: wheel build, py3.10/3.11/3.12 tests, cheap-import, ruff check and ruff format.
>
> Still open, and both need a human: publishing 0.1.0 to PyPI, and the container rebuild in section 2.
> Until 0.1.0 is published, do **not** remove `packages/spatialqc/` from this repository — the module `environment.yml` pins `spatialqc==0.1.0` from PyPI, which does not exist yet, so the in-tree copy is still the only working source.

The package was developed in-tree so the existing test suite could gate every step.
Extract it with `git subtree split`, which rewrites the commits touching that subdirectory so its contents sit at the repository root:

Split to a **fresh branch name every time** and push that.
A subtree split is not incremental: re-splitting after new commits produces different SHAs, so a branch left over from an earlier split is stale the moment anything lands in `packages/spatialqc`.

```bash
# Pick a name that has not been used before -- never reuse an old split branch.
STAMP=$(date +%Y%m%d%H%M)
git subtree split --prefix=packages/spatialqc -b "spatialqc-export-$STAMP"
git push https://github.com/altos-labs/spatialqc.git "spatialqc-export-$STAMP:main"
```

Do **not** resurrect a branch called `spatialqc-standalone`.
One existed from the first split and was already behind the published `main`; pushing it would have been rejected, and force-pushing it would have rolled back the device-scoping and `max_gpus` fixes.

This has been run and verified: the resulting branch has `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, `src/`, `tests/` and `.github/workflows/ci.yml` at its root, with both threshold YAMLs and `py.typed` present, and the CI workflow became active once it was a repository root.

One caveat to set expectations: `subtree split` filters by path, so the extracted branch carries only the commits that touched `packages/spatialqc` — four, at the time of writing.
The scripts' earlier history under `bin/` does **not** follow, because that is a different path.
If the full provenance of `image_qc.py` matters in the new repository, use `git filter-repo` with a path rename instead:

```bash
git filter-repo --path bin/image_qc.py --path bin/snr_metrics.py \
    --path bin/transcript_qc_processing.py --path bin/transcript_stream.py \
    --path packages/spatialqc --path-rename packages/spatialqc/:
```

Otherwise keep this repository as the historical record and let the new one start from the extraction point.

After extraction, in this repository:

- Remove `packages/spatialqc/` and rely on the published wheel.
- Keep the exact pin in both module `environment.yml` files, and bump it in the same commit as any container tag bump.
- Point `.github/workflows/python-tests.yml` at the published package, or keep a submodule/checkout step if PRs should be able to test unreleased package changes.

Publishing:

```bash
cd packages/spatialqc
python -m build
python -m zipfile -l dist/*.whl | grep -E 'yaml|py.typed'   # package data must be present
twine upload dist/*
```

## 4. Follow-ups, in priority order

### 4.1 Threshold YAML now exists in three places

`bin/roi_image_qc_thresholds.yaml` (staged by the pipeline and read by the report notebook), `packages/spatialqc/src/spatialqc/data/image_qc_thresholds.yaml` (the packaged default), and `_EMBEDDED_THRESHOLDS_FALLBACK` inside `bin/transcript_qc.qmd`.

They are identical today and will drift.
The packaged copy should become authoritative, with the pipeline obtaining the path from the package rather than shipping its own.
Note also that `transcript_qc_thresholds.yaml` is read only by the report notebook — the Python analysis never opens it — and that both notebooks resolve a hardcoded `conf/`-relative fallback path although the files live in `bin/`.

### 4.2 The 26 GPU dispatch sites are not yet on the Backend protocol

`spatialqc.backend` is the designed interface and is used by the new code, but the ported analysis still dispatches internally in three different styles: a `_get_backend` tuple factory, direct `cp.` calls under `cp.cuda.Device(gpu_id)`, and array-type inference via `_is_device_array` / `_device_or_host` with no flag threaded through.

Unifying them is a semantic change to numerical code and needs a real-data regression suite to validate, which does not exist.
It should be done incrementally, one dispatch style at a time, each with before/after metrics on a real bundle.

`--device` is honoured today at the single point where the analysis decides which GPUs to use (`resolve_available_gpus`), which is sufficient to select and enforce a mode.

> **Correction 2026-09-09, after GPU validation.**
> The paragraph above is right about *mode selection* and wrong to imply the dispatch styles are merely untidy.
> It was written by a session with no GPU, so the multi-GPU path had never run.
>
> These sites are **not** device-scoped, and that is a live defect on any instance with more than one GPU, not a migration concern:
>
> - Tiles are sharded round-robin across devices — `gpu_ids[slot % len(gpu_ids)]` (`image/qc.py:3215`), and again at 3789 and 3908 — so tiles genuinely land on devices 1..N.
> - `_process_tile_for_consumers` takes `keep_mean_device` / `keep_focus_device` and returns arrays still resident on that device.
> - Those arrays reach the `consume()` callbacks, which run raw CuPy operations on them — device-array indexing at 2296-2298, `cp.asnumpy` at 2988, `roi_snr_db_batch(st, xp=cp)` at 2997. `grep` confirms **no `cp.cuda.Device` context exists anywhere between lines 2200 and 3100**, and line 2941 (`isinstance(array, cp.ndarray)`) shows these callbacks knowingly accept device arrays.
> - Reading a device array while another device is current is an unrecoverable `cudaErrorIllegalAddress` that **aborts the interpreter** rather than raising, so the Nextflow task dies with no Python traceback. Reproduced on a 4x L4 host; see F1 in `docs/reviews/2026-09-09_REVIEW_spatialqc-port-gpu.md`.
>
> Why it has not been seen: image QC runs under `label 'process_gpu_qc'` and has in practice received one GPU, where every device id is 0 and the bug is unreachable.
>
> The `CupyBackend` half of this is fixed (commit `3c97443`); the raw sites in `image/qc.py` are not.
> **Audit them before any multi-GPU deployment, or pin the analysis to a single device until the migration lands.**
> The cheapest interim guard is to cap `resolve_available_gpus` at one device, which costs throughput but cannot abort.

### 4.3 The memory instrumentation is a divergent fork, deliberately left alone

`_log_mem`, `_log_mem_summary`, `_MEM_PEAK` and the `_CGROUP_*` readers exist in both analyses and are **not** copies: the image version logs through `logging` and tracks four figures including a process-tree RSS, the transcript version prints and tracks two, and `_log_mem_summary` shares no lines between them.

They were not merged, for two reasons: the fork is real rather than accidental, and the image QC test suite monkeypatches these names on the module object (`_CGROUP_SOURCES`, `_CGROUP_PEAK_PATHS`, `_cgroup_peak`, `_MEM_PEAK`, `_PLANE_SPILL_BYTES`).
Moving them to a shared module and re-exporting would leave those patches pointing at the wrong object, turning roughly 18 assertions into silent no-ops that read the real `/sys/fs/cgroup` and pass for the wrong reason.

If they are ever merged, the patches must be re-pointed in the same commit.

### 4.4 Remaining mechanical modernization

`ruff check --select UP,I,F --fix` over the ported modules resolves ~110 findings, almost all typing syntax (`Dict` to `dict`, `Optional[X]` to `X | None`).
This is semantically inert but touches many lines, so it belongs in its own commit, applied only with the full suite green before and after.
The `C408` findings (`dict()` calls that could be literals) and `UP035` (deprecated `typing` imports) are worth reviewing by hand.

## 5. What the Quarto notebooks do and do not constrain

Both report notebooks import **no** project modules — no `sys.path` manipulation, no `importlib`, no `exec`.
They read output files only.

The frozen contract is therefore the set of **output filenames and JSON key names**, not the Python layout.
That is what made this restructure safe, and it is what any future refactor must preserve.
