# spatialqc: extraction to a standalone repository, and follow-ups

Status: the package exists and is wired into the pipeline inside this repository.
It is not yet published, and the QC containers do not yet contain it.
This document covers the remaining steps and the decisions that need a human.

## 1. Decision required: the GPU focus-score change

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

Recommended: re-run image QC on two or three calibration samples and compare the reported blurry-tile percentages against the YAML cutoffs before release.

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

The package was developed in-tree so the existing test suite could gate every step.
Extract it with `git subtree split`, which rewrites the commits touching that subdirectory so its contents sit at the repository root:

```bash
git subtree split --prefix=packages/spatialqc -b spatialqc-standalone
git push git@github.com:altos-labs/spatialqc.git spatialqc-standalone:main
```

This has been run on `feat/spatialqc-package` and verified: the resulting branch has `pyproject.toml`, `README.md`, `LICENSE`, `.gitignore`, `src/`, `tests/` and `.github/workflows/ci.yml` at its root, with both threshold YAMLs and `py.typed` present, and the CI workflow becomes active once it is a repository root.

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
