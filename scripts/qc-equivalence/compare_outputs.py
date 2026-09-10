#!/usr/bin/env python3
"""Compare two image-QC output trees for exact numerical equivalence.

Usage: compare_outputs.py REF_DIR PORT_DIR

Reports, in order:
  0. Sanity assertions -- proves the runs produced real content, not two
     identically-empty trees (a "no diff" verdict on empty output is worthless).
  1. File-set differences (present in one tree only).
  2. JSON differences, every numeric leaf compared exactly.
  3. Table differences (parquet / csv / csv.gz): row count, column set, and
     per-column exact-match plus max|delta| for numerics.
  4. Figure differences: existence and pixel dimensions only. Raw PNG bytes
     differ from matplotlib metadata, so byte equality is not required.

Volatile keys (timestamps / versions / paths) are bucketed separately, but a
volatile-bucketed key whose value is NUMERIC and differs is reported as
SUSPECT and counted as a failure. A broad substring like "date" or "dir" can
match a real metric name, and a real numeric change hidden in that bucket would
otherwise read as a clean run. Only non-numeric volatile differences are
treated as benign.

Exit code is 0 only when no numeric difference is found and every sanity
assertion passes.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VOLATILE_SUBSTRINGS = (
    "timestamp",
    "date",
    "time",
    "runtime",
    "elapsed",
    "duration",
    "version",
    "path",
    "dir",
    "hostname",
    "host",
    "outdir",
    "created",
    "generated",
    "uuid",
    "seconds",
)

TABLE_SUFFIXES = (".parquet", ".csv", ".csv.gz", ".tsv")
FIGURE_SUFFIXES = (".png", ".jpg", ".jpeg", ".pdf", ".svg")

# Independently known-good values for this bundle, from the GPU baseline.
# These make the verdict a positive statement about content, not just "no diff".
# Structural invariants that must hold on BOTH sides for a comparison to mean
# anything. Keys are FULLY QUALIFIED and matched exactly: an earlier version
# matched on a dotted suffix, which silently compared the 2-D baseline value
# against `blur_gmm_1d.rois_blurred_gmm` and reported a bogus MISMATCH.
# Structural invariants that must hold on BOTH sides before a "no differences"
# verdict means anything. These are deliberately NOT expected values: an earlier
# version hardcoded R2_control's counts (712236 ROIs, 126045 cells), which made
# every run on a different bundle report a scary MISMATCH and a "NOT identical"
# verdict even when the two sides agreed on every single number. The magic
# constants were checking which bundle you ran, not whether the comparison was
# valid.
#
# What actually has to be true is: the key exists, it is non-trivial (so two
# empty runs cannot pass by agreeing on nothing), and the two sides agree.
SANITY = [
    ("roi_qc_metrics.json", "total_rois"),
    ("image_qc_metrics.json", "total_cells"),
]

# Values from the GPU baseline. These legitimately differ between the CPU and
# GPU code paths (different algorithms), so they are reported for information
# and never counted as failures.
BASELINE_INFO = [
    ("roi_qc_metrics.json", "blur_gmm_2d.rois_blurred_gmm", 487910),
    (
        "roi_qc_metrics.json",
        "blur_gmm_2d.pct_blurred_gmm_tissue_filtered",
        34.69194064438745,
    ),
    ("roi_qc_metrics.json", "blur_gmm_1d.total_rois_tissue_filtered", 343489),
]


def is_volatile(dotted_key: str) -> bool:
    low = dotted_key.lower()
    return any(sub in low for sub in VOLATILE_SUBSTRINGS)


def is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten(obj, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}[{index}]"))
    else:
        out[prefix] = obj
    return out


def leaves_equal(a, b) -> bool:
    """Exact equality, with NaN==NaN equal (both mean 'not computed')."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b or a == b
    if is_number(a) and is_number(b):
        if (
            isinstance(a, float)
            and isinstance(b, float)
            and math.isnan(a)
            and math.isnan(b)
        ):
            return True
        return a == b
    return a == b


def read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith(".csv.gz"):
        with gzip.open(path, "rt") as handle:
            return pd.read_csv(handle)
    if name.endswith(".tsv"):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def figure_dims(path: Path):
    if path.suffix.lower() in (".pdf", ".svg"):
        return None
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return f"unreadable: {type(exc).__name__}"


def compare_json(ref: Path, port: Path):
    """Return (hard_findings, suspect_numeric_volatile, benign_volatile)."""
    ref_flat = flatten(json.loads(ref.read_text()))
    port_flat = flatten(json.loads(port.read_text()))

    hard: list[str] = []
    suspect: list[str] = []
    benign: list[str] = []

    for key in sorted(set(ref_flat) - set(port_flat)):
        hard.append(f"  KEY MISSING IN PORT: {key} (ref={ref_flat[key]!r})")
    for key in sorted(set(port_flat) - set(ref_flat)):
        hard.append(f"  KEY ADDED IN PORT:   {key} (port={port_flat[key]!r})")

    for key in sorted(set(ref_flat) & set(port_flat)):
        a, b = ref_flat[key], port_flat[key]
        if leaves_equal(a, b):
            continue
        line = f"  {key}: ref={a!r} port={b!r}"
        if not is_volatile(key):
            hard.append(line)
        elif is_number(a) and is_number(b):
            # A numeric value under a volatile-looking name may still be a real
            # metric ("date"/"dir"/"time" are broad substrings). Do not absolve it.
            suspect.append(line)
        else:
            benign.append(line)
    return hard, suspect, benign


def compare_table(ref: Path, port: Path, findings: list[str]) -> None:
    try:
        ref_df, port_df = read_table(ref), read_table(port)
    except Exception as exc:  # noqa: BLE001
        findings.append(f"  UNREADABLE ({type(exc).__name__}: {exc})")
        return

    if len(ref_df) != len(port_df):
        findings.append(f"  ROW COUNT: ref={len(ref_df)} port={len(port_df)}")

    only_ref = [c for c in ref_df.columns if c not in port_df.columns]
    only_port = [c for c in port_df.columns if c not in ref_df.columns]
    if only_ref:
        findings.append(f"  COLUMNS MISSING IN PORT: {only_ref}")
    if only_port:
        findings.append(f"  COLUMNS ADDED IN PORT:   {only_port}")

    if len(ref_df) != len(port_df):
        return

    for col in [c for c in ref_df.columns if c in port_df.columns]:
        a, b = ref_df[col], port_df[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            av, bv = a.to_numpy(), b.to_numpy()
            if av.dtype.kind == "f" and bv.dtype.kind == "f":
                both_nan = np.isnan(av) & np.isnan(bv)
            else:
                both_nan = np.zeros(len(av), dtype=bool)
            diff = (av != bv) & ~both_nan
            n = int(diff.sum())
            if n:
                with np.errstate(invalid="ignore"):
                    max_abs = float(
                        np.nanmax(
                            np.abs(av[diff].astype(float) - bv[diff].astype(float))
                        )
                    )
                findings.append(
                    f"  COL {col}: {n}/{len(av)} rows differ, max|delta|={max_abs:.6g}"
                )
        else:
            neq = int((a.astype(str) != b.astype(str)).sum())
            if neq:
                findings.append(
                    f"  COL {col}: {neq}/{len(a)} non-numeric values differ"
                )


def run_sanity(port_files: dict, ref_files: dict) -> tuple[bool, list[str]]:
    """Assert the outputs contain real, expected content on BOTH sides."""
    lines: list[str] = []
    ok = True
    cache: dict[str, dict] = {}

    def load(side_files, name):
        key = f"{id(side_files)}:{name}"
        if key not in cache:
            match = [r for r in side_files if r.name == name]
            cache[key] = (
                flatten(json.loads(side_files[match[0]].read_text())) if match else {}
            )
        return cache[key]

    for filename, key in SANITY:
        values = {}
        for label, files in (("ref", ref_files), ("port", port_files)):
            flat = load(files, filename)
            if not flat:
                lines.append(f"  MISSING {filename} on {label} side")
                ok = False
                continue
            # Exact key match only -- a suffix match conflates
            # blur_gmm_1d.X with blur_gmm_2d.X.
            if key not in flat:
                lines.append(f"  {label}: {filename} has no key {key!r}")
                ok = False
                continue
            values[label] = flat[key]

        if len(values) != 2:
            continue
        ref_value, port_value = values["ref"], values["port"]
        if not (isinstance(ref_value, (int, float)) and ref_value > 0):
            # Two runs that both produced nothing would otherwise "agree".
            lines.append(
                f"  {filename}:{key} = {ref_value!r} - not a non-trivial count,"
                " so agreement below proves nothing"
            )
            ok = False
        elif ref_value != port_value:
            lines.append(
                f"  *** {filename}:{key} DISAGREES  ref={ref_value!r} port={port_value!r}"
            )
            ok = False
        else:
            lines.append(f"  ok  {filename}:{key} = {ref_value!r} on both sides")

    # GPU-baseline values: informational, never a failure. The CPU and GPU code
    # paths are different algorithms, so these are expected to differ between
    # them; what matters is that ref and port agree with EACH OTHER, which the
    # sections below test.
    for filename, key, baseline in BASELINE_INFO:
        for label, files in (("ref", ref_files), ("port", port_files)):
            flat = load(files, filename)
            if key in flat:
                note = (
                    "matches baseline"
                    if flat[key] == baseline
                    else "differs from baseline"
                )
                lines.append(
                    f"  [info] {label}: {key} = {flat[key]!r} ({note} {baseline!r})"
                )

    for label, files in (("ref", ref_files), ("port", port_files)):
        flat = load(files, "image_qc_metrics.json")
        if flat:
            top = {k.split(".")[0] for k in flat}
            lines.append(
                f"  [info] {label}: image_qc_metrics.json has {len(top)} top-level keys"
            )

    # snr_metrics.json must have a verdict at all.
    for label, files in (("ref", ref_files), ("port", port_files)):
        flat = load(files, "snr_metrics.json")
        hits = [k for k in flat if k.endswith("overall_snr_verdict")]
        if hits:
            lines.append(f"  {label}: snr_metrics.json:{hits[0]} = {flat[hits[0]]!r}")
        elif flat:
            lines.append(f"  {label}: snr_metrics.json has NO overall_snr_verdict")
            ok = False

    # Informational: CPU vs the GPU baseline figure.
    for label, files in (("ref", ref_files), ("port", port_files)):
        flat = load(files, "roi_qc_metrics.json")
        hits = [k for k in flat if k.endswith("pct_blurred_gmm_tissue_filtered")]
        if hits:
            lines.append(
                f"  {label}: {hits[0]} = {flat[hits[0]]!r} "
                f"(GPU baseline 34.69194064438745; a float-rounding gap here is "
                f"CPU-vs-GPU reduction order, NOT a ref-vs-port defect)"
            )
    return ok, lines


def main() -> int:
    ref_root, port_root = Path(sys.argv[1]), Path(sys.argv[2])

    ref_files = {p.relative_to(ref_root): p for p in ref_root.rglob("*") if p.is_file()}
    port_files = {
        p.relative_to(port_root): p for p in port_root.rglob("*") if p.is_file()
    }

    numeric_problem = False

    print("=" * 78)
    print(f"REF : {ref_root}  ({len(ref_files)} files)")
    print(f"PORT: {port_root}  ({len(port_files)} files)")
    print("=" * 78)

    print("\n## 0. SANITY (are these real outputs at all?)")
    sane, lines = run_sanity(port_files, ref_files)
    for line in lines:
        print(line)
    if not sane:
        numeric_problem = True
        print(
            "  *** sanity assertions FAILED - a 'no diff' verdict below would be meaningless"
        )

    print("\n## 1. FILE-SET DIFFERENCES")
    only_ref = sorted(set(ref_files) - set(port_files))
    only_port = sorted(set(port_files) - set(ref_files))
    if only_ref or only_port:
        for rel in only_ref:
            print(f"  ONLY IN REF : {rel}")
        for rel in only_port:
            print(f"  ONLY IN PORT: {rel}")
        numeric_problem = True
    else:
        print("  none - identical file sets")

    shared = sorted(set(ref_files) & set(port_files))

    print("\n## 2. JSON METRICS")
    json_clean = True
    all_benign: list[str] = []
    for rel in [r for r in shared if r.suffix == ".json"]:
        hard, suspect, benign = compare_json(ref_files[rel], port_files[rel])
        if hard:
            json_clean = False
            numeric_problem = True
            print(f"\n  [DIFF] {rel}")
            for line in hard:
                print(line)
        if suspect:
            json_clean = False
            numeric_problem = True
            print(f"\n  [SUSPECT - numeric value under a volatile-looking key] {rel}")
            for line in suspect:
                print(line)
        if benign:
            all_benign.append(f"  [volatile, non-numeric] {rel}")
            all_benign.extend(benign)
    if json_clean:
        print("  no numeric differences in any shared JSON")
    if all_benign:
        print("\n  -- volatile non-numeric differences (read these, do not skip) --")
        for line in all_benign:
            print(line)

    print("\n## 3. TABLES")
    table_clean = True
    tables = [r for r in shared if str(r).lower().endswith(TABLE_SUFFIXES)]
    for rel in tables:
        findings: list[str] = []
        compare_table(ref_files[rel], port_files[rel], findings)
        if findings:
            table_clean = False
            numeric_problem = True
            print(f"\n  [DIFF] {rel}")
            for line in findings:
                print(line)
    if not tables:
        print("  no shared tables")
    elif table_clean:
        print(f"  all {len(tables)} shared table(s) match exactly")

    print("\n## 4. FIGURES (dimensions only; byte differences are expected)")
    figs = [r for r in shared if r.suffix.lower() in FIGURE_SUFFIXES]
    dim_mismatch = 0
    for rel in figs:
        a, b = figure_dims(ref_files[rel]), figure_dims(port_files[rel])
        if a != b:
            dim_mismatch += 1
            print(f"  [DIM DIFF] {rel}: ref={a} port={b}")
    print(f"  {len(figs)} shared figure(s), {dim_mismatch} with differing dimensions")

    print("\n" + "=" * 78)
    if numeric_problem:
        print("VERDICT: NOT numerically identical - see differences above.")
    else:
        print("VERDICT: numerically identical across all shared JSONs and tables,")
        print("         and both trees pass the content sanity assertions.")
    print("=" * 78)
    return 1 if numeric_problem else 0


if __name__ == "__main__":
    sys.exit(main())
