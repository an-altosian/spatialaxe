# Review: spatialqc extraction and GPU code paths

Reviewed: `git diff eb56727..HEAD` on `feat/spatialqc-package` (commits `cb20364`, `62615ac`, `0e483ec`, `5443487`, `df4920e`).
Hardware used for verification: 1-2 of 4x NVIDIA L4, driver 580.159.03, CuPy 14.0.1 / SciPy 1.16.2 / NumPy 2.2.6.
Accepted items from `docs/plans/2026-09-09_PLAN_spatialqc-extraction-and-followups.md` (§2 unpublished containers/wheel, §4.2 dispatch not unified, §4.3 forked memory instrumentation, §4.4 remaining ruff modernization) are not re-reported.

## Verdict

**No blocking defect on the wired production path.** The refactor is genuinely a move: at AST level (docstrings stripped) only 19 of the 214 image-QC definitions changed and every one is cosmetic or the intended `--device` addition; SNR and transcript-stream are pure typing modernization; the frozen contract (output filenames, JSON key names) is preserved. The one numerical change — the fused Laplacian-of-Gaussian on the GPU — is **CONFIRMED correct on real hardware**: CPU and GPU now agree to float32 rounding (max relative deviation 5.1e-07 at the production `lap_sigma=1.0`), and the whole suite (244 passed, 1 skipped) including the `gpu`-marked parity tests passes on an L4.

**However, the never-executed new GPU code in `spatialqc/backend.py` has two confirmed defects that its own test suite structurally cannot see.** Neither affects a run today, because `image/qc.py` does not import `spatialqc.backend` (§4.2). Both must be fixed *before* anyone starts the §4.2 migration, because the first one breaks the exact multi-GPU sharding pattern the class docstring prescribes, and the second silently inverts the meaning of `--max-gpus 0`.

Recommendation: land the refactor; fix F1 and F2 in a follow-up commit before any dispatch-site migration begins. Neither is a release blocker for the pipeline as wired.

---

## Findings

### F1 — HIGH (API contract; zero production impact today) — `CupyBackend` filter methods run outside the device context, so every non-current device fails

`packages/spatialqc/src/spatialqc/backend.py:254`, `:257`, `:260` (`uniform_filter`, `laplace`, `gaussian_laplace`)

`__init__` (`:245`) stores `self._device = _cp.cuda.Device(device_id)` and `to_device` (`:247-249`) correctly enters `with self._device:` — so the array *lands* on `device_id`. The three filter methods then call the `cupyx` kernels with no device context at all, so they launch on whatever the *current* device is. Any `CupyBackend(device_id=N)` where `N` is not the current device therefore fails on its first filter call.

Failure scenario: the class docstring states "Multi-GPU runs create one backend per device and shard tiles across them" — exactly this. `get_backend("gpu", device_id=1)` produces the same broken object.

**CONFIRMED.** With `CUDA_VISIBLE_DEVICES=0,1`, one backend per device, `to_device` then `gaussian_laplace`:

```
device 0: OK
device 1: FAILED ValueError: The device where the array resides (1) is different from
          the current device (0). Peer access is unavailable between these devices.
-> 1 ok / 1 failed
```

`uniform_filter` and `laplace` raise identically. Wrapping the calls in `with cp.cuda.Device(i):` at the call site makes both devices succeed, which isolates the cause to the missing context.

Why the tests miss it: every GPU-marked test in `tests/test_backend.py` constructs `CupyBackend()` with the default `device_id=0`. No test ever passes a non-zero `device_id`, so a single-GPU *or* multi-GPU runner both pass. CPU-only CI skips these entirely.

Fix direction: wrap each of the three methods in `with self._device:` (as `to_device` already does), and add a test parameterised over `available_gpu_ids()` — skipping when fewer than two devices are visible — that round-trips a tile through each device's backend.

### F2 — MEDIUM (latent, becomes HIGH on the §4.2 migration; inverts a documented CLI contract) — `max_gpus=0` means "no GPUs" in `backend.available_gpu_ids` but "no cap" in `qc.resolve_available_gpus`

`packages/spatialqc/src/spatialqc/backend.py:100-131` (`ids = ids[:max_gpus]` at `:131`) vs `packages/spatialqc/src/spatialqc/image/qc.py:992`

Two functions added in the same diff implement "the `--max-gpus` cap", carry the same AWS-Batch-accelerator rationale in their docstrings, and disagree on `0`:

| Call | Result |
| ---------------------------------------- | -------- |
| `backend.available_gpu_ids(max_gpus=0)` | `[]` |
| `qc.resolve_available_gpus("auto", 0)` | `[0, 1]` |

`resolve_available_gpus` guards the truncation with `if available_gpus and max_gpus and ...`, so falsy `0` skips the cap; `available_gpu_ids` slices unconditionally, and `ids[:0]` is empty. The `resolve_available_gpus` docstring states "`0` and `None` mean 'no cap'"; the `available_gpu_ids` docstring does not mention `0` at all. The click option `--max-gpus` **defaults to 0** and is documented "0 = use every device detected" (`image/qc.py:13555`), and `tests/test_backend.py::test_available_gpu_ids_respects_the_cap` asserts the *opposite* convention (`== []`), enshrining the divergence.

Failure scenario: the §4.2 migration replaces the `resolve_available_gpus` call at `qc.py:12627` with `backend.available_gpu_ids(max_gpus)`. Every default run (`--max-gpus 0`) then silently becomes CPU-only on a GPU node — a ~50x slowdown that produces a correct-looking result, i.e. exactly the failure mode `--device gpu` was introduced to prevent.

**CONFIRMED** (both values observed above on the L4 box with two devices visible).

Fix direction: pick one convention and make `available_gpu_ids` normalise it (`if max_gpus:` rather than `if max_gpus is not None:`), then fix `test_available_gpu_ids_respects_the_cap` to match. Document `0` in both docstrings.

### F3 — LOW — `CupyBackend.__init__` does not validate `device_id`, so an out-of-range device fails lazily and obscurely

`packages/spatialqc/src/spatialqc/backend.py:237-245`; `get_backend` at `:283-310`

`get_backend("gpu", device_id=N)` checks only that `available_gpu_ids()` is non-empty — it never checks that `N` is *in* that list. `_cp.cuda.Device(N)` construction does not validate either, so the object is built successfully and the first `to_device` raises a raw CUDA error far from the mistake.

**CONFIRMED** (2 devices visible, `device_id=6`):

```
constructed: CupyBackend(device_id=6)
to_device RAISED CUDARuntimeError cudaErrorInvalidDevice: invalid device ordinal
```

Fix direction: in `get_backend`, assert `device_id in ids` and raise the same style of explanatory `RuntimeError` the no-device branch already raises.

### F4 — LOW/MEDIUM (process) — the two module `environment.yml` pip blocks run pip's resolver, and the "already satisfied" claim is asserted rather than enforced

`modules/local/image_qc/environment.yml:53`, `modules/local/transcript_qc/environment.yml:38`

`.github/workflows/python-tests.yml` deliberately installs with `pip install -e packages/spatialqc --no-deps`, with a comment saying that letting pip resolve the extras "would shadow those pins with PyPI builds". The two container `environment.yml` files then do the opposite — `- spatialqc[image,image-regionprops,spatial-stats]==0.1.0` and `- spatialqc[transcript]==0.1.0` inside a plain `pip:` block, with pip's resolver active — while asserting in a comment that conda has already satisfied everything.

I cross-checked every `Requires-Dist` extracted from the built wheel's METADATA against the conda pins in both files. **Today the claim holds** for every scientific dependency (details in "Verified clean" below), with one gap:

- `click>=8.1` is a **core** spatialqc dependency and is *not* an explicit conda pin in `modules/local/transcript_qc/environment.yml`. It is present only transitively (papermill/jupyter), so pip will either accept the transitive build or pull click from PyPI. Harmless in itself (pure Python, no ABI surface), but it means the block is not in fact "all satisfied".

The real exposure is structural: nothing enforces the invariant. A future conda pin drifting below a floor — `pyarrow` back below 18 in transcript QC (already pinned at 20.0.0 rather than 21 because `anndata` caps it), or `zarr` moving to 3.x in image QC against `zarr>=2.18,<3` — would make pip silently replace a conda build inside the container, with no error and no test that would notice.

**PLAUSIBLE** (verified by pin-vs-metadata inspection; not executed, since `spatialqc==0.1.0` is unpublished — §2).

Fix direction: either add `--no-deps` to the pip block (nf-core `environment.yml` accepts a bare `--no-deps` entry ahead of the requirements, or use a `requirements.txt` with the flag), or add an explicit `conda-forge::click=...` pin to transcript QC and a CI check that the wheel's `Requires-Dist` floors are all met by the conda pins.

### F5 — LOW — `--device cpu` is now emitted on every default run and hard-disables opportunistic GPU use

`conf/modules.config:419`, `modules/local/image_qc/main.nf:128`

`ext.device = { params.use_gpu && (params.image_qc_gpus as int) > 0 ? 'gpu' : 'cpu' }`, and `params.use_gpu` defaults to `false` (`nextflow.config:32`). Previously the script always autodetected. In practice this changes little on AWS Batch: a task with no `accelerator` directive gets no CUDA devices mounted into the container, so opportunistic GPU use was never available there. Where it does bite is a local `-profile docker` run on a GPU workstation, which previously took the GPU streaming path and now takes the CPU path — and the CPU branch of `calculate_roi_focusscore` (`qc.py:5140-5165`) decodes every channel at full resolution into host RAM and computes whole-image maps, instead of folding per-tile reductions the way the streaming path does (streaming is gated on `_use_gpu` at `qc.py:5020`). On a multi-gigapixel sample that is a large host-RAM increase.

This is the deliberate trade the module comment describes, and it is the safer default. Flagging it only so it is a decision rather than a surprise: confirm no launch profile relies on opportunistic detection. `tower_launch/` contains no `use_gpu` setting, so any Tower launch that wants GPU image QC must now set it explicitly.

Note the same change **fixes** a pre-existing bug: `params.image_qc_gpus = 0`, documented as "0 = CPU only" (`nextflow.config:118`), previously produced `--max-gpus 0` = "no cap", so on a GPU node it used *every* GPU. It now correctly resolves to `--device cpu`.

**PLAUSIBLE** (reasoned from config; not exercised through Nextflow).

### F6 — LOW — the shared helpers were rewritten, not copied, contrary to the comments claiming a verified byte-identical move

`packages/spatialqc/src/spatialqc/bundle.py`, `versions.py`, `stats.py`; comments at `image/qc.py:94-97` and `transcript/qc.py:26-31`

Both call sites say the vendored blocks "were verified identical by diff before being replaced with this import". The blocks that were *removed* were indeed identical to each other, but the code that replaced them is a rewrite: locals renamed, `_parse_xenium_version` → `parse_xenium_version`, `_tool_label` → `tool_label`, a new `_load_experiment_json` extracted, `open(...).write()` → `Path.write_text(encoding="utf-8")`, control flow restructured. The comment invites a future reader to trust the move without re-reading it.

I compared all ten helpers function-by-function against the originals. **Behaviour is preserved**, with one intentional-looking improvement: base `read_xenium_analysis_sw_version` had `meta.get(...)` *inside* the try, so an `experiment.xenium` whose top level is a JSON array raised an uncaught `AttributeError`; `_load_experiment_json` now returns `None` via an `isinstance(metadata, dict)` check. Strictly more robust, and unreachable on a real bundle.

Fix direction: reword the comments to "reimplemented with identical semantics; see review" so the claim matches the code.

### F7 — LOW — `read_xenium_pixel_size_um` now exists twice, and the copy in `bundle.py` is dead

`packages/spatialqc/src/spatialqc/bundle.py:112` and `packages/spatialqc/src/spatialqc/image/snr.py:1788`

`bundle.py` exports it in `__all__`; nothing imports it (`image/qc.py` imports only `XENIUM_PIXEL_SIZE_UM`, `read_xenium_analysis_sw_version`, `read_xenium_major_version`, `resolve_segmentation_software`). The live copy is `snr.py`'s. The two differ only in the `isinstance(metadata, dict)` guard described in F6, so they agree on every real bundle — but the whole point of `bundle.py` was to end this duplication, and a two-copy pixel-size reader is precisely the drift the module docstring warns about. Also relevant to the §4.1 TODO in `xenium_image_qc_report.qmd:913`, which wants `pixel_size_um` emitted into `roi_qc_metrics.json`.

Fix direction: have `snr.py` import from `spatialqc.bundle` and delete its copy.

### F8 — LOW — `thresholds.py` is a complete, tested, exported module that nothing uses, and its `path=None` semantics differ from the live loader's

`packages/spatialqc/src/spatialqc/thresholds.py` (165 lines, covered by `tests/test_thresholds.py`)

Not imported by `image/qc.py`, `transcript/qc.py`, `snr.py`, or `spatialqc/__init__.py`. It is the intended §4.1 landing point, so its existence is fine — but the semantic gap should be recorded now: `load_image_qc_thresholds(None)` returns the **packaged YAML's** `image_qc` section, whereas the live path (`--roi-thresholds-yaml` unset) uses the analysis's **hardcoded in-code defaults** and never reads a YAML at all. Adopting `thresholds.py` as a drop-in for the current loader would therefore change which thresholds a default run uses. `channel_config` also *raises* `KeyError` on a populated-but-incomplete `channels:` section, where the current loader falls back to defaults.

Fix direction: when §4.1 is done, land the switch with a before/after comparison of every resolved threshold on a real sample, not as a mechanical import swap.

### F9 — TRIVIAL — documentation and metadata drift in the new files

- `packages/spatialqc/src/spatialqc/__init__.py:13` — "Both accept a `device` selector"; `run_transcript_qc` does not, and `transcript/__init__.py` correctly says so.
- `packages/spatialqc/pyproject.toml:53` — comment cites `spatialqc.image.io`, which does not exist.
- `packages/spatialqc/pyproject.toml:157` — mypy comment cites `utils.py`, which does not exist.
- `packages/spatialqc/src/spatialqc/backend.py:47` — `sanitize` (defined at `:319`) is omitted from `__all__` although `tests/test_backend.py` imports it by name.
- `packages/spatialqc/src/spatialqc/backend.py:91-98` — `gpu_is_available` docstring says it "additionally queries the CUDA runtime" so a container with CuPy but no GPU is "correctly reported as unavailable". It only calls `getDeviceCount()`; a present-but-unusable device still counts. Same limitation as `detect_gpu_ids` (`image/qc.py:922`), so not a regression, but the docstring overstates it.
- Comments in both `environment.yml` files and `snr.py` still reference `bin/image_qc.py`, `transcript_qc_processing.py`, `transcript_stream.py` as if they existed. No *executable* reference to a removed script remains anywhere (checked across `Makefile`, `Dockerfile*`, `*.nf`, `*.config`, `*.yml`, `*.toml`, `*.json`, `*.py`, `*.qmd`), and `bin/` at HEAD correctly no longer contains `image_qc.py`, `snr_metrics.py`, `transcript_qc_processing.py`, `transcript_stream.py`, or `bin/tests/`.

---

## Verified clean

Each of these was checked and found correct; those marked (executed) were demonstrated by running code on the L4 box.

**The fused Laplacian-of-Gaussian correction (the one numerical change) — (executed).** Replicating `compute_laplacian_variance_map`'s exact float32→LoG→float64→uniform_filter→variance chain on a 512x512 synthetic edge+noise tile, on both backends:

| `lap_sigma` | max rel. deviation of the LoG response | correlation | focus-map max rel. | var ratio |
| ----------- | -------------------------------------- | ----------- | ------------------ | --------- |
| 0.5 | 2.1e-07 | 1.0000000000 | 2.7e-07 | 1.000000 |
| **1.0** | **3.5e-07** | **1.0000000000** | **5.1e-07** | **1.000000** |
| 2.0 | 3.3e-07 | 1.0000000000 | 9.9e-07 | 1.000000 |

`cupyx.scipy.ndimage.gaussian_laplace` exists in CuPy 14.0.1 (the pinned version) and matches `scipy.ndimage.gaussian_laplace` operator-for-operator; both default to `mode='reflect'`, `truncate=4.0`, and both return float32 for float32 input. The plan's §1 claim is therefore sound, and the caveat that sample-level blurry-tile percentage cutoffs were calibrated against shim numbers stands — recalibration before release remains the right call.

**Dtype/precision seams in the GPU focus paths.** `compute_laplacian_variance_map` (`qc.py:1086-1167`, the LoG path; the divergent operator lived here) and the Laplacian phase of `_compute_channel_maps_on_gpu` (`qc.py:1276-1290`) are byte-unchanged from `eb56727` apart from the operator swap, and are symmetric: both sides cast the input to float32, apply the LoG, promote to float64 before the `E[X²]-E[X]²` subtraction, clamp at zero, and cast back to float32. No missing `.get()` / `cp.asnumpy()`, no device array handed to a scipy/numpy function.

**Device contexts in the production GPU code.** Every `cp.` call in `_compute_channel_maps_on_gpu`, `_process_tile_on_gpu`, `_stream_channels`, `compute_ccfs_map` and `compute_laplacian_variance_map` is inside `with cp.cuda.Device(gpu_id):`. (F1 is confined to `backend.py`, which the analysis does not import.)

**GPU-OOM fallback cannot mix scales within a sample.** The only CPU fallback is in `compute_all_focus_maps` (`qc.py:3929-3951`), and it recomputes *all* channels from scratch on the CPU and sets `use_gpu = False` for the remainder — it does not splice per-channel GPU results with per-channel CPU results. The streaming/tiled path has no per-tile fallback at all. Combined with the operator fix, a sample is now graded on exactly one scale.

**`--device` is honoured at a single, effective gate — (executed, via the suite).** `resolve_available_gpus` (`qc.py:953-1000`) is the only decision point; `--device cpu` returns `[]`, which leaves `use_gpu=False` and `gpu_ids=None` at the `calculate_roi_focusscore` call site (`qc.py:12665`), so the internal `detect_gpu_ids() or [0]` at `qc.py:5024` — the one place that could bypass the selector — is unreachable on a CPU run. `--device gpu` with no device raises rather than falling back. 14 tests in `test_device_resolution.py` cover this and pass.

**Refactor fidelity — AST-level, docstrings stripped.**
- `image/qc.py` vs `bin/image_qc.py`: 19 changed definitions. 8 are `'Quoted'` → bare annotations on `spawn`/`merge`; 4 are moving a heavy import (`napari_skimage_regionprops`, `napari_simpleitk_image_processing`, `mpl_toolkits`) from module scope into the function that uses it, plus one `open(p,'r')` → `open(p)`; `save_versions_file` gains an `import nsitk` inside its existing try/except so the `napari-simpleitk-image-processing` version still resolves; `__MODULE_LEVEL__` is import reordering plus the shim removal plus the `spatialqc.bundle` import; `main` → `run_image_qc` differs only by the `device` parameter, the `resolve_available_gpus` call, and `sorted(list(g))` → `sorted(g)`. Removed definitions are exactly the six helpers that moved to `bundle.py`. **No other behaviour change.**
- `image/snr.py` vs `bin/snr_metrics.py`: 22 changed definitions, **all** `Dict`/`List`/`Optional`/`Tuple` → PEP 585/604 syntax. Zero logic change.
- `transcript/stream.py` vs `bin/transcript_stream.py`: **zero** changed definitions.
- `transcript/qc.py` vs `bin/transcript_qc_processing.py`: the analysis body is unchanged (it still reads `args.<name>`); the change is `main()` split into `_build_parser()` / `main(argv)` / `run_transcript_qc(args)` with a `TranscriptQCOptions` dataclass. The `add_argument` calls are identical to the originals, flag for flag, including the `nargs` on `--non-gene-prefix` / `--stain-names` / `--seg-versions-file`.

**The frozen contract (output filenames and JSON keys).** Two independent checks agree. (a) The set of string literals in each ported module was extracted and diffed against `eb56727`: **zero added literals** in `transcript/qc.py` and `snr.py`; in `image/qc.py` the only additions are `--device`, `cpu`, `gpu` and two log strings, and the only removals are the segmentation labels and `experiment.xenium`/`analysis_sw_version` that moved to `bundle.py`. (b) More decisively, the AST comparison shows that none of the 19 changed image definitions and neither of the 2 changed transcript definitions contains output-writing code, so no f-string filename could have changed either (the literal extraction alone would not have caught an f-string). `versions.yml` key order and content in `save_versions_file` (`qc.py:8743-8756`) and `dump_versions` are byte-preserved, including `click` and the `pathlib: built-in` entry.

**Packaging — (executed).** Built the wheel with `python -m build --wheel --no-isolation`. Contents confirm: `spatialqc/py.typed` present, both `spatialqc/data/image_qc_thresholds.yaml` (20952 B) and `spatialqc/data/transcript_qc_thresholds.yaml` (17981 B) present, `entry_points.txt` = `spatialqc-image-qc = spatialqc.image.qc:main` / `spatialqc-transcript-qc = spatialqc.transcript.qc:main`, `dist-info/licenses/LICENSE` included, version `0.1.0` resolved dynamically from `spatialqc.__version__`. The repository-root `.gitignore` has a blanket `data/` rule that would have swallowed the threshold YAMLs; `packages/spatialqc/.gitignore` re-includes them and `git check-ignore` confirms they are not ignored — both are tracked (37 tracked files total; the local `build/` tree is correctly ignored).

**No import-time side effects — (executed).** `import spatialqc` pulls in only `typing`, `re`, `enum` and friends — **zero** heavy modules (no numpy, pandas, pyarrow, click, scipy, matplotlib, cupy). `spatialqc.__version__` resolves without any extra installed, so the modules' `python3 -c 'import spatialqc; print(spatialqc.__version__)'` version-topic evals work in a bare environment. The PEP 562 lazy exports in `__init__.py`, `image/__init__.py` and `transcript/__init__.py` all raise `AttributeError` for unknown names rather than importing blindly.

**Dependency pins.** Every `Requires-Dist` floor in the built wheel is met by the conda pins in the corresponding module `environment.yml` (image QC: numpy 2.3.3, pandas 2.3.2, pyarrow 21.0.0, pyyaml 6.0.3, click 8.3.0, scipy 1.16.2, scikit-image 0.25.2, scikit-learn 1.7.2, tifffile 2025.9.20, imagecodecs 2025.8.2, zarr 2.18.7 — inside the `<3` cap, numba 0.62.0, h5py 3.14.0, matplotlib 3.10.6, seaborn 0.13.2, napari-simpleitk-image-processing 0.4.9, napari-skimage-regionprops 0.10.1, esda 2.9.0, libpysal 4.14.1; transcript QC: scanpy 1.11.4, anndata 0.12.2, pyarrow 20.0.0 satisfying `>=18` while respecting anndata's `<21` cap). The extras split is coherent: `gpu` is never a hard dependency, and the `image` extra's `zarr<3` matches the code's zarr-2 Group API usage. See F4 for the one gap (`click` in transcript QC) and the absent enforcement.

**Nextflow wiring — (executed).** Every flag the two modules emit is parsed, and every parsed parameter is accepted:
- Image QC: introspecting `click` and `inspect.signature(run_image_qc)` gives **zero** click params not accepted by `run_image_qc` and **zero** `run_image_qc` params never supplied by the CLI (21 each). All 17 flags built in `modules/local/image_qc/main.nf` — including the new `--device` — map to real click options. `--stain-names` is a click option the module never passes, which is intentional (it retains the CLI flag while `a5241ec` dropped the pipeline param) and harmless.
- Transcript QC: `TranscriptQCOptions(**vars(args))` was executed with the full argv the module builds. The 10 argparse dests and the 10 dataclass fields are an exact set match, so neither a stray dest nor a missing field can raise at startup. All 8 flags in `modules/local/transcript_qc/main.nf` are parsed.
- Flags parsed but not acted on (the defect class this project has been bitten by before). Three exist, and **all three are byte-identical to `eb56727` — none was introduced by this diff**: `--threads` (transcript) binds to `TranscriptQCOptions.threads` and is never read, the real knob being `NUMBA_NUM_THREADS` exported in `main.nf:100`; `--num-row-groups` (transcript) is parsed and then unconditionally forced to `None` with an explanatory log line (`transcript/qc.py:474-481`), because the streaming aggregation reads the whole file at bounded memory; `--stain-names` is accepted by both CLIs and passed by neither module — in image QC it *is* consumed (`qc.py:12532-12546`, semicolon-split with a default list), so it remains a working option, just an unused one. Nothing is passed-but-unparsed in either direction.
- `ext.device` in `conf/modules.config:419` parses correctly under Groovy precedence (`&&` binds before `?:`) and is null-safe (`null as int` → 0 → `'cpu'`). The `task.ext.device ?: 'auto'` fallback is dead in practice since the closure always returns a non-empty string, which is fine.
- Both modules gained a `spatialqc` version topic channel and both container tags moved to `2.0.0`, consistent with §2.

**Test suite — (executed).** `pytest tests/` in `packages/spatialqc` on the GPU host: **244 passed, 1 skipped**, no failures. The `gpu`-marked cross-backend parity tests (`uniform_filter` at sizes 3/5/9, `laplace`, `gaussian_laplace` at sigma 0.5/1.0/2.0, round-trip, synchronize) actually executed rather than skipping, and pass at `rtol=1e-5`. `sanitize` was verified to work on a device array too (`np.isfinite` dispatches through CuPy's `__array_ufunc__`; NaN and ±Inf are zeroed in place, returning a `cupy.ndarray`). `tests/conftest.py` correctly contains no `sys.path` manipulation, which is what makes the suite safe under `pytest-xdist`.
