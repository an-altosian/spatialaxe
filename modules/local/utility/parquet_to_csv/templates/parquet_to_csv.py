#!/usr/bin/env python3
"""
Convert a Parquet file to CSV format.

Reads a Parquet file and writes it as CSV, optionally gzip-compressed.
"""

from pathlib import Path

import pandas as pd

# Nextflow-injected variables
TRANSCRIPTS = "${transcripts}"
EXTENSION = "${extension}"
PREFIX = "${prefix}"


def convert_parquet(
    transcripts: str,
    extension: str = ".csv",
    prefix: str = "",
) -> None:
    """
    Convert a Parquet file to CSV or CSV.GZ format.

    Args:
        transcripts: Filename of the input parquet file
        extension: Output extension ('.csv' or '.gz' for gzip)
        prefix: Output directory prefix
    """
    df = pd.read_parquet(transcripts, engine="pyarrow")

    Path(prefix).mkdir(parents=True, exist_ok=True)

    if extension == ".gz":
        output = transcripts.replace(".parquet", ".csv.gz")
        df.to_csv(f"{prefix}/{output}", compression="gzip", index=False)
    else:
        output = transcripts.replace(".parquet", ".csv")
        df.to_csv(f"{prefix}/{output}", index=False)

    return None


if __name__ == "__main__":
    convert_parquet(
        transcripts=TRANSCRIPTS,
        extension=EXTENSION,
        prefix=PREFIX,
    )
