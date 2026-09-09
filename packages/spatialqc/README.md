# spatialqc

Quality-control analytics for 10x Genomics Xenium spatial transcriptomics runs.

`spatialqc` grades a Xenium output bundle along two independent axes and emits metrics JSON, tabular
data and figures that a report can render. It began as two standalone scripts inside the
[`nf-core/spatialaxe`](https://github.com/nf-core/spatialaxe) Nextflow pipeline and was extracted into
a package so that the analysis is installable, importable, testable and versionable on its own.

| Analysis       | Entry point            | What it grades                                                                                                           |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Image QC       | `run_image_qc`         | Morphology-image focus and contrast maps, per-tile blur classification, stain intensity, signal-to-noise, per-cell texture |
| Transcript QC  | `run_transcript_qc`    | Per-cell and per-gene count distributions, negative-control probe rates, cell-assignment statistics                       |

Both analyses run on either a CUDA GPU or the CPU, selected explicitly.

## Installation

The scientific stack is split into extras so a consumer solves only for what it uses.

```bash
pip install 'spatialqc[image]'        # image QC
pip install 'spatialqc[transcript]'   # transcript QC
pip install 'spatialqc[all]'          # both
pip install 'spatialqc[all,gpu]'      # both, with CuPy for CUDA 12
```

| Extra               | Pulls in                                                        |
| ------------------- | --------------------------------------------------------------- |
| `image`             | scipy, scikit-image, scikit-learn, tifffile, zarr, numba, h5py, matplotlib, seaborn |
| `image-regionprops` | napari-simpleitk-image-processing, napari-skimage-regionprops (per-cell texture only) |
| `spatial-stats`     | esda, libpysal (Moran's I for the negative-probe SNR metric)     |
| `transcript`        | scipy, scanpy, anndata, h5py, matplotlib, seaborn                |
| `gpu`               | cupy-cuda12x                                                     |
| `dev`               | pytest, pytest-xdist, ruff, mypy, build                          |

`import spatialqc` itself needs only numpy, pandas, pyarrow, pyyaml and click: the heavy submodules
load on first attribute access, so importing the package does not pay for a stack you did not ask for.

## Quick start

### As a library

```python
from pathlib import Path

from spatialqc import run_image_qc, run_transcript_qc

image_result = run_image_qc(
    xenium_bundle_dir=Path("/data/output-XETG00001__0001234__sample_a"),
    outdir=Path("qc/image"),
    sample_id="sample_a",
    device="auto",          # "cpu" | "gpu" | "auto"
    roi_size=35,
)

transcript_result = run_transcript_qc(
    xenium_bundle_dir=Path("/data/output-XETG00001__0001234__sample_a"),
    outdir=Path("qc/transcript"),
    threads=8,
)
```

### As a CLI

```bash
spatialqc-image-qc \
    --xenium-bundle-dir /data/output-XETG00001__0001234__sample_a \
    --outdir qc/image \
    --sample-id sample_a \
    --device auto

spatialqc-transcript-qc \
    --xenium-bundle-dir /data/output-XETG00001__0001234__sample_a \
    --outdir qc/transcript \
    --threads 8
```

## CPU and GPU modes

Backend selection is resolved once, in `spatialqc.backend.get_backend`, rather than branched at each
call site. Each backend uses native primitives — `cupyx.scipy.ndimage` on the GPU, `scipy.ndimage`
plus Numba reduction kernels on the CPU — so neither path is a degraded mirror of the other.

```python
from spatialqc.backend import available_gpu_ids, get_backend

get_backend("cpu")     # NumpyBackend
get_backend("gpu")     # CupyBackend, or RuntimeError if no CUDA device is usable
get_backend("auto")    # GPU when available, CPU otherwise
available_gpu_ids(max_gpus=1)
```

`device="gpu"` **raises** when no CUDA device is usable rather than falling back. A silent fallback
returns a correct-looking result at roughly 50x the cost on a node that was requested precisely because
it had a GPU. `auto` is for interactive use; automated callers should pass `cpu` or `gpu` explicitly.

Host/device transfers happen at the edges (`to_device` / `to_numpy`), once per tile.

## Thresholds

Both analyses ship a default threshold YAML as package data, read through `importlib.resources`, so the
package is self-contained when installed from a wheel. A caller can override it:

```python
from spatialqc.thresholds import load_image_qc_thresholds

defaults = load_image_qc_thresholds()                     # packaged default
tuned = load_image_qc_thresholds(Path("my_thresholds.yaml"))  # caller override
```

## Development

```bash
pip install -e '.[dev,all]'
pytest -q
ruff check src tests
ruff format --check src tests
mypy src/spatialqc
```

Tests import the installed package — there is no `sys.path` manipulation in `conftest.py`, which is why
the editable install is required. GPU parity tests are marked `gpu` and skip automatically when CuPy is
unavailable:

```bash
pytest -m "not gpu"    # CPU-only host
```

Build and verify a wheel actually contains the package data:

```bash
python -m build
python -m zipfile -l dist/*.whl | grep -E 'yaml|py.typed'
```

## Use from a Nextflow pipeline

Pin an exact version in the module `environment.yml` so runs stay reproducible:

```yaml
dependencies:
  - pip
  - pip:
      - spatialqc[image,gpu]==0.1.0
```

Report the version dynamically through the versions topic channel — never hardcode it:

```groovy
tuple val("${task.process}"), val('spatialqc'),
    eval("python3 -c 'import spatialqc; print(spatialqc.__version__)'"), topic: versions
```

## License

MIT. See [LICENSE](LICENSE).
