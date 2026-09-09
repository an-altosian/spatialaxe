---
name: python-packaging
description: Use when creating, restructuring, or publishing an installable Python package - pyproject metadata, src layout, extras, entry points, package data, optional GPU/accelerator backends, version single-sourcing, editable installs, wheel publishing
category: capability_uplift
version: 1.0.0
eval_path: .claude/evals/python-packaging.json
---

# Python Packaging Skill

Companion to `/python`, which covers in-package code style (constants, TypedDict, exceptions, fixtures).
This skill covers the **distribution boundary**: how a package is named, built, installed, imported, and consumed.

Invoke when the task is "turn these scripts into a package", "publish this", "why can't my tests import it",
"add an optional GPU dependency", or "ship a data file with the package".

## RULE P0: Distribution Name and Import Name Must Be Coherent

The single most common packaging defect. Three names exist and must be chosen deliberately:

| Name              | Where it lives                              | Example           |
| ----------------- | ------------------------------------------- | ----------------- |
| Distribution name | `[project] name` — what `pip install` takes | `spatial-qc`      |
| Import package    | `src/<dir>/` — what `import` takes          | `spatialqc`       |
| Console script    | `[project.scripts]` key                     | `spatialqc-image` |

Convention: distribution name in kebab-case, import name in snake_case, same word.
`pip install spatial-qc` -> `import spatialqc`.

**Never** let them drift apart (`name = "spatialbench"` shipping `import xenium_helpers`). Symptoms:
a stray `<other>.egg-info/`, `pip uninstall <what-you-import>` failing, and consumers unable to pin the
right thing. If a rename is unavoidable because downstream code imports the old name, ship an explicit
shim module that re-exports and warns — do not leave the mismatch undocumented.

```python
# src/spatialqc/_compat/xenium_helpers.py — bridge for pre-rename consumers
"""Deprecated alias for :mod:`spatialqc`. Kept so pinned pipelines keep importing."""

import warnings

from spatialqc import utils  # noqa: F401

warnings.warn("xenium_helpers is renamed to spatialqc", DeprecationWarning, stacklevel=2)
```

## RULE P1: src Layout, Always

```
pyproject.toml
src/<import_name>/__init__.py
tests/
```

`src/` makes it **impossible** to import the package without installing it. That is the point: a flat
layout silently imports from the working directory, so tests pass locally and the wheel is broken
(missing subpackage, missing data file). With `src/`, `pip install -e .` is mandatory and CI tests the
real installed artifact.

Corollary: **`sys.path.insert(...)` in `conftest.py` is a packaging smell.** It means the code is not
installable. Replace it with `pip install -e .` and delete the hack. A `conftest.py` that manipulates
`sys.path` also makes test collection order load-bearing (a module importable only when a sibling ran
first), which fails under `pytest-xdist`.

## RULE P2: Single-Source the Version

Declare once, read everywhere. Never maintain a version in both `pyproject.toml` and `__init__.py`.

```toml
[project]
name = "spatial-qc"
dynamic = ["version"]

[tool.setuptools.dynamic]
version = { attr = "spatialqc.__version__" }
```

At runtime prefer installed metadata, so a wheel reports what pip resolved:

```python
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spatial-qc")
except PackageNotFoundError:  # running from a source tree
    __version__ = "0.0.0.dev0"
```

Expose it for pipeline version reporting: `python3 -c "import spatialqc; print(spatialqc.__version__)"`.

## RULE P3: Core Dependencies Thin, Extras for Everything Else

A dependency in `[project] dependencies` is installed for **every** consumer. Put there only what
`import <package>` itself needs. Everything feature-specific goes in an extra.

```toml
dependencies = ["numpy>=2.0", "pandas>=2.2", "pyarrow>=18"]

[project.optional-dependencies]
image = ["scikit-image>=0.24", "tifffile>=2024.9", "zarr>=2.18,<3"]
transcript = ["scanpy>=1.10", "anndata>=0.11"]
gpu = ["cupy-cuda12x>=13"]
dev = ["pytest>=7", "pytest-xdist>=3", "ruff>=0.9", "mypy>=1"]
all = ["spatial-qc[image,transcript]"]
```

Audit before shipping: every declared dependency must be `grep`-able as an actual import. A package
that pins `harmonypy`, `opencv-python`, and `loompy` but imports none of them forces a multi-GB solve
on every consumer and breaks container builds on unrelated conflicts.

Version bounds: floor on what you use (`>=`), upper bound **only** for a known incompatibility, and
comment why. `zarr>=2.18,<3  # 3.x changed the group API used by open_zarr` is useful; a bare `<3` is
cargo cult.

## RULE P4: Heavy and Optional Imports Go Inside Functions

Top-level imports run on `import <package>`. A module-scope `import scanpy` costs seconds and makes
the whole package unimportable when an extra is absent — which is why test suites end up stubbing
`sys.modules`.

```python
def render_umap(adata_path: Path) -> Path:
    import scanpy as sc  # local: only the `transcript` extra needs this
    ...
```

Guard truly optional accelerators once, in one module, and let the failure be explicit:

```python
try:
    import cupy as cp

    HAS_CUPY = True
except Exception:  # broad: a CUDA-less host raises more than ImportError
    cp = None
    HAS_CUPY = False
```

**No silent fallback.** If the caller asked for GPU and there is no GPU, raise — a run that quietly
drops to CPU produces a correct-looking result at 50x the cost and nobody notices.

## RULE P5: Dual CPU/GPU Backends via a Protocol

When a package supports both, make selection explicit and one-time, not scattered `if use_gpu:`.

```python
from typing import Literal, Protocol


class Backend(Protocol):
    name: str
    xp: ModuleType  # numpy or cupy

    def to_device(self, a: NDArray) -> Any: ...
    def to_numpy(self, a: Any) -> NDArray: ...
    def uniform_filter(self, a: Any, size: int) -> Any: ...
    def synchronize(self) -> None: ...


def get_backend(device: Literal["auto", "cpu", "gpu"]) -> Backend:
    if device == "gpu":
        if not HAS_CUPY:
            raise RuntimeError("device='gpu' requested but CuPy is unavailable")
        return CupyBackend()
    if device == "auto":
        return CupyBackend() if HAS_CUPY else NumpyBackend()
    return NumpyBackend()
```

Rules that keep both paths genuinely fast rather than one being a crippled mirror:

- Each backend uses its **native** primitives — `cupyx.scipy.ndimage` on GPU, `scipy.ndimage` plus
  `numba` kernels and a thread pool on CPU. Do not emulate one in the other.
- Write array-module-generic kernels as `f(x, xp)` where the caller passes `numpy`/`cupy`, so a single
  implementation compiles to both.
- Keep host<->device transfers at the edges. Convert once per tile, not per operation.
- Expose `--device {auto,cpu,gpu}` on the CLI. `auto` is for humans; pipelines pass an explicit value
  so a missing driver fails loudly instead of silently costing a day of compute.
- **Parity test** is mandatory: same synthetic input through both backends, assert
  `np.allclose(..., rtol=1e-5)`, `@pytest.mark.skipif(not HAS_CUPY)`. It is the only evidence the two
  modes agree.

## RULE P6: Ship Data Files as Package Data, Read via importlib.resources

A threshold YAML, a reference table, or a notebook template must travel inside the wheel. Never resolve
it relative to `__file__` or assume a repo checkout.

```toml
[tool.setuptools.package-data]
spatialqc = ["data/*.yaml", "py.typed"]
```

```python
from importlib.resources import files


def default_thresholds() -> dict:
    text = files("spatialqc.data").joinpath("image_qc_thresholds.yaml").read_text()
    return yaml.safe_load(text)
```

Pattern: packaged default, caller override. `load_thresholds(path=None)` returns the packaged YAML when
`path` is None, so the package is self-contained but the pipeline can still pass a tuned file.

## RULE P7: Console Scripts, Not Executable Modules

```toml
[project.scripts]
spatialqc-image = "spatialqc.cli.image_qc:main"
spatialqc-transcript = "spatialqc.cli.transcript_qc:main"
```

`pip install` puts these on `PATH` on every platform. This replaces the "copy `script.py` into the
container and `chmod 755`" pattern, and the mode-755 trap that goes with it.

Keep the CLI layer **thin**: parse, validate, call one library function, exit. Anything else belongs in
the library, where it is importable and testable. The test for a CLI is a call to the library function;
the test for the CLI itself is that `--help` works.

When replacing existing scripts in a pipeline, keep flag names **byte-identical** so the caller changes
one token (`image_qc.py` -> `spatialqc-image`) and nothing else. Renaming flags in the same change makes
a mechanical port indistinguishable from a behavior change.

## RULE P8: Declare a Typed Public API

```python
# src/spatialqc/__init__.py
from spatialqc.image import run_image_qc
from spatialqc.transcript import run_transcript_qc

__all__ = ["run_image_qc", "run_transcript_qc", "__version__"]
```

- `__all__` is the contract. Anything not in it is private and may change.
- Add an empty `src/spatialqc/py.typed` marker so consumers get your annotations (PEP 561).
- Prefix internals with `_`. A module named `_resources.py` is not part of the API.
- Public functions take and return explicit types — `Path`, not `str`; a `TypedDict`, not a bare `dict`.

## RULE P9: Test the Installed Artifact

```bash
pip install -e '.[dev,image,transcript]'
pytest -q                      # imports the installed package, no path games
python -m build                # sdist + wheel
python -c "import spatialqc; print(spatialqc.__version__)"
```

Verify the wheel actually contains what you think — a missing data file or subpackage is invisible until
a consumer installs it:

```bash
python -m zipfile -l dist/*.whl | grep -E 'yaml|py.typed'
```

## Consuming a Package from a Nextflow / container pipeline

- Pin an exact version in `environment.yml` (`- pip: ["spatial-qc==0.2.0"]`) so runs are reproducible.
  A floating dependency makes a pipeline non-reproducible even when the pipeline commit is fixed.
- Report the version dynamically through the versions topic channel — never hardcode it:

  ```groovy
  tuple val("${task.process}"), val('spatialqc'),
      eval("python3 -c 'import spatialqc; print(spatialqc.__version__)'"), topic: versions
  ```

- Provide a source-tree override for CI so a PR tests its own code, not the baked-in wheel:
  `PYTHONPATH` / an editable install in the CI job, guarded by one env var.
- Bump the pinned version and the container tag in the same commit as the package release.

## Anti-Patterns

| Anti-pattern                                 | Why it breaks                                                     |
| -------------------------------------------- | ----------------------------------------------------------------- |
| `sys.path.insert()` in `conftest.py`         | Package is not installable; collection order becomes load-bearing |
| Dist name != import name                     | Consumers cannot pin or uninstall; stray `egg-info`               |
| Version in both pyproject and `__init__`     | They drift; wheel reports the wrong one                           |
| Every dependency in `[project] dependencies` | Multi-GB solves, unrelated conflicts, broken container builds     |
| `import scanpy` at module scope              | Slow import; unimportable without the extra; forces test stubs    |
| Silent CPU fallback when GPU is missing      | Correct-looking result at 50x cost, undetected                    |
| Data file found via `__file__` / `../..`     | Works in a checkout, `FileNotFoundError` from a wheel             |
| `setup.py` alongside `pyproject.toml`        | Two sources of truth; deprecated invocation paths                 |
| `python script.py` as the entry point        | Not on `PATH`; needs mode 755; no dependency resolution           |
| Restructuring and rewriting in one commit    | A mechanical move becomes unreviewable and drift is undetectable  |

## Porting Scripts into a Package - Order of Operations

Follow this sequence; it is what keeps a large port reviewable.

1. **Establish a green baseline.** Run the existing tests first. Without them there is no safety net.
2. **Scaffold** `pyproject.toml` + `src/<pkg>/` + `pip install -e .`.
3. **Move files verbatim.** One script -> one module, imports rewritten to package imports. No logic edits.
4. **Port the tests**, deleting the `sys.path` hack. Get green. This is the checkpoint that proves the
   move was faithful.
5. **Only then split** modules along real seams, re-running tests after each split. Module-level
   globals (locks, caches, `HAS_*` flags) are where a careless split changes behavior.
6. **Mechanical modernization** last: `ruff check --select UP,I,F --fix` (import order, dead imports,
   typing syntax). Semantically inert, visibly idiomatic.
7. **Do not hand-rewrite numerical function bodies** during a port. Without a real-data regression
   suite, a semantic rewrite cannot be validated — state that limit explicitly rather than doing it
   quietly.
