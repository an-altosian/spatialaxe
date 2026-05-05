#!/usr/bin/env python3
"""
Convert a segmentation mask TIFF to uint32 dtype.

XeniumRanger import-segmentation requires uint32 masks, but upstream
segmenters (e.g. StarDist) often emit int32 labels. This script reads
the input mask, casts it to uint32, and writes the result.
"""

import numpy as np
import tifffile

# Nextflow-injected variables
INPUT_PATH = "${mask}"
OUTPUT_PATH = "${prefix}_uint32_mask.tif"


def convert_mask_to_uint32(input_path: str, output_path: str) -> None:
    """
    Read a mask TIFF, cast to uint32, and write to output_path.

    Args:
        input_path: Path to input mask TIFF (any integer dtype).
        output_path: Path where the uint32 mask will be written.
    """
    mask = tifffile.imread(input_path)
    print(f"Input dtype: {mask.dtype}, shape: {mask.shape}, labels: {mask.max()}")
    tifffile.imwrite(output_path, mask.astype(np.uint32))
    print("Output dtype: uint32")


if __name__ == "__main__":
    convert_mask_to_uint32(input_path=INPUT_PATH, output_path=OUTPUT_PATH)
