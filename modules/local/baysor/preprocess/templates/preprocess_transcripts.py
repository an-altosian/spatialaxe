#!/usr/bin/env python3
"""
Preprocess Xenium transcripts for Baysor segmentation.

Filters transcripts based on quality score and spatial coordinate thresholds,
removes negative control probes, and outputs filtered CSV for Baysor compatibility.
"""

import os

import pandas as pd

# Nextflow-injected variables
TRANSCRIPTS = "${transcripts}"
PREFIX = "${prefix}"
MIN_QV = "${min_qv}"
MIN_X = "${min_x}"
MAX_X = "${max_x}"
MIN_Y = "${min_y}"
MAX_Y = "${max_y}"


def filter_transcripts(
    transcripts: str,
    min_qv: float = 20.0,
    min_x: float = 0.0,
    max_x: float = 24000.0,
    min_y: float = 0.0,
    max_y: float = 24000.0,
    prefix: str = "",
) -> None:
    """
    Filter transcripts based on the specified thresholds.

    Args:
        transcripts: Path to transcripts parquet file
        min_qv: Minimum Q-Score to pass filtering
        min_x: Minimum x-coordinate threshold
        max_x: Maximum x-coordinate threshold
        min_y: Minimum y-coordinate threshold
        max_y: Maximum y-coordinate threshold
        prefix: Output directory prefix
    """
    df = pd.read_parquet(transcripts, engine="pyarrow")

    # filter transcripts df with thresholds, ignore negative controls
    filtered_df = df[
        (df["qv"] >= min_qv)
        & (df["x_location"] >= min_x)
        & (df["x_location"] <= max_x)
        & (df["y_location"] >= min_y)
        & (df["y_location"] <= max_y)
        & (~df["feature_name"].str.startswith("NegControlProbe_"))
        & (~df["feature_name"].str.startswith("antisense_"))
        & (~df["feature_name"].str.startswith("NegControlCodeword_"))
        & (~df["feature_name"].str.startswith("BLANK_"))
    ]

    # change cell_id of cell-free transcripts to "0" (Baysor's no-cell sentinel).
    # Modern Xenium stores cell_id as a string ("UNASSIGNED" for cell-free transcripts);
    # legacy Xenium used integer -1. Normalize to string and handle both cases — pandas 3
    # rejects mixing int values into a string-dtype column.
    filtered_df["cell_id"] = filtered_df["cell_id"].astype(str)
    neg_cell_row = filtered_df["cell_id"].isin(["-1", "UNASSIGNED"])
    filtered_df.loc[neg_cell_row, "cell_id"] = "0"

    # Output filtered transcripts as CSV for Baysor 0.7.1 compatibility.
    # Baysor's Julia Parquet.jl cannot read modern pyarrow Parquet files
    # (pyarrow 15+ writes size_statistics Thrift field 16 unconditionally,
    # which Baysor's old Thrift deserializer doesn't recognize).
    os.makedirs(prefix, exist_ok=True)
    filtered_df.to_csv(f"{prefix}/filtered_transcripts.csv", index=False)

    return None


def main() -> None:
    """
    Run preprocess transcripts as nf module.
    """
    filter_transcripts(
        transcripts=TRANSCRIPTS,
        min_qv=float(MIN_QV),
        min_x=float(MIN_X),
        max_x=float(MAX_X),
        min_y=float(MIN_Y),
        max_y=float(MAX_Y),
        prefix=PREFIX,
    )

    return None


if __name__ == "__main__":
    main()
