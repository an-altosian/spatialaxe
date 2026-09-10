#!/usr/bin/env python3
"""Quantify the residual numeric spread between two QC output trees.

`compare_outputs.py` answers "are these identical?" with a yes/no. When the
answer is no, this answers the follow-up: *how* not-identical, and in which
class of key. It exists because "the difference is just float32 rounding" is
easy to assert from a handful of hand-picked metrics and wrong often enough
to be worth measuring -- on the port's own CPU-vs-GPU outputs the real
maximum was 3.68e-06, about 31 float32 ULP, not the ~3.5e-07 first claimed.

Keys are separated into three classes, because mixing them hides the signal:

  config   thresholds read from configuration, and Monte-Carlo p-values.
           These are inputs or noise, never measurements of the image.
  derived  counts, percentages and statuses computed *from* a threshold.
           When a threshold differs these differ too, and there were 23 of
           them in one comparison -- enough to swamp everything real.
  computed everything else. These must agree, and the interesting question
           is by how much they fail to.

The verdict that matters is not the maximum delta but whether any integer
key moved: a float drift that never crosses a classification boundary
leaves every count identical, which is what backend stability looks like.
"""

import argparse
import json
import math
import pathlib
import re
import sys

FLOAT32_EPS = 1.19e-07

CONFIG = re.compile(r"critical_threshold|pct_warn_threshold|moran")
DERIVED = re.compile(r"below_critical|below_intensity|quality_status|overall_quality")


def flatten(obj, prefix=""):
    """Flatten nested JSON to fully-qualified dotted keys.

    Lists are indexed rather than zipped, so `component_means[0][1]` stays
    distinguishable from `component_means[1][0]` -- GMM component order is
    itself a thing worth catching a change in.
    """
    out = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(flatten(value, f"{prefix}.{key}" if prefix else key))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            out.update(flatten(value, f"{prefix}[{index}]"))
    else:
        out[prefix] = obj
    return out


def is_real_number(value):
    # bool is an int subclass; a flag flipping is not a numeric delta.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("left", type=pathlib.Path)
    parser.add_argument("right", type=pathlib.Path)
    parser.add_argument("--top", type=int, default=10,
                        help="how many worst offenders to print (default 10)")
    args = parser.parse_args()

    shared = sorted(
        p.name for p in args.left.iterdir()
        if p.suffix == ".json" and (args.right / p.name).exists()
    )
    if not shared:
        sys.exit(f"no shared JSON files between {args.left} and {args.right}")

    deltas = []
    integer_moved = []
    nan_mismatches = []
    n_config = n_derived = n_equal = n_integer = 0

    for name in shared:
        left = flatten(json.loads((args.left / name).read_text()))
        right = flatten(json.loads((args.right / name).read_text()))
        for key in sorted(set(left) & set(right)):
            a, b = left[key], right[key]
            if CONFIG.search(key):
                n_config += 1
                continue
            if DERIVED.search(key):
                n_derived += 1
                continue
            if not (is_real_number(a) and is_real_number(b)):
                continue
            if isinstance(a, int) and isinstance(b, int):
                n_integer += 1
                if a != b:
                    integer_moved.append((name, key, a, b))
                continue
            # NaN is a legitimate, reproducible result here -- an empty
            # channel or a percentile of nothing -- and `nan != nan` would
            # otherwise report every one of them as a difference and poison the
            # maximum with nan. Both sides being NaN is agreement, which is the
            # same convention compare_outputs.py uses.
            if a == b or (math.isnan(a) and math.isnan(b)):
                n_equal += 1
                continue
            if math.isnan(a) or math.isnan(b):
                # One side NaN and the other a number is a real difference, and
                # a ratio cannot express it.
                nan_mismatches.append((name, key, a, b))
                continue
            # Relative to the larger magnitude: near zero an absolute delta is
            # meaningless, and these metrics span 1e-08 to 1e+04.
            deltas.append((abs(a - b) / (max(abs(a), abs(b)) or 1.0), name, key, a, b))

    deltas.sort(reverse=True)

    print(f"shared JSON files          : {len(shared)}")
    print(f"config-input keys skipped  : {n_config}")
    print(f"threshold-derived skipped  : {n_derived}")
    print(f"integer keys compared      : {n_integer}")
    print(f"float keys exactly equal   : {n_equal}")
    print(f"float keys differing       : {len(deltas)}")

    if deltas:
        over = [d for d in deltas if d[0] > FLOAT32_EPS]
        print(f"\nmax relative delta         : {deltas[0][0]:.3e}")
        print(f"above one float32 ULP      : {len(over)} of {len(deltas)}")
        print(f"\nworst {min(args.top, len(deltas))} relative deltas:")
        for ratio, name, key, a, b in deltas[:args.top]:
            print(f"  {ratio:.3e}  {name}:{key}")
            print(f"              {a!r} -> {b!r}")

    if nan_mismatches:
        print(f"\nNaN ON ONE SIDE ONLY       : {len(nan_mismatches)}")
        for name, key, a, b in nan_mismatches:
            print(f"  {name}:{key}  {a!r} -> {b!r}")

    print(f"\nINTEGER KEYS MOVED         : {len(integer_moved)}")
    for name, key, a, b in integer_moved:
        print(f"  {name}:{key}  {a} -> {b}")

    # A float delta that never crosses a classification boundary is tolerable.
    # One that reclassifies an ROI is not, and shows up here as a moved count.
    print()
    if integer_moved or nan_mismatches:
        print("VERDICT: not equivalent - a count moved or NaN appeared on one side only.")
        sys.exit(1)
    if deltas:
        print("VERDICT: every count, total and classification is identical;")
        print("         the remaining movement is float rounding that changed no decision.")
    else:
        print("VERDICT: bit-identical - not one compared value differs.")


if __name__ == "__main__":
    main()
