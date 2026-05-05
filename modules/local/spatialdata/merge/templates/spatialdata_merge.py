#!/usr/bin/env python3
"""Merge two spatialdata bundles to create a layered spatialdata object."""

import json
import os
import shutil

import spatialdata  # noqa: F401  (kept so versions topic via `import spatialdata` is valid)

# Nextflow-injected variables
RAW_BUNDLE = "${raw_bundle}"
REDEFINED_BUNDLE = "${redefined_bundle}"
PREFIX = "${prefix}"
OUTPUT_FOLDER = "${outputfolder}"


def main():
    """Run spatialdata merge."""
    print("[START]")

    output_dir = f"spatialdata/{PREFIX}/{OUTPUT_FOLDER}"

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    for root, _, files in os.walk(RAW_BUNDLE):
        rel_path = os.path.relpath(root, RAW_BUNDLE)
        target_path = os.path.join(output_dir, rel_path)
        os.makedirs(target_path, exist_ok=True)
        for file in files:
            shutil.copy(os.path.join(root, file), os.path.join(target_path, file))

    for category in ["points", "shapes", "tables"]:
        category_path = os.path.join(output_dir, category)
        if os.path.exists(category_path):
            for folder in next(os.walk(category_path))[1]:
                old_path = os.path.join(category_path, folder)
                print(folder)
                new_path = os.path.join(category_path, f"raw_{folder}")
                os.rename(old_path, new_path)

    for category in ["points", "shapes", "tables"]:
        add_category_path = os.path.join(REDEFINED_BUNDLE, category)
        output_category_path = os.path.join(output_dir, category)
        os.makedirs(output_category_path, exist_ok=True)

        if os.path.exists(add_category_path):
            for folder in next(os.walk(add_category_path))[1]:
                src_folder = os.path.join(add_category_path, folder)
                dest_folder = os.path.join(output_category_path, f"redefined_{folder}")
                shutil.copytree(src_folder, dest_folder)

    # Invalidate consolidated metadata in zarr.json -- the directory renames above
    # made the element paths in the metadata stale. Without consolidated metadata,
    # sd.read_zarr() discovers elements by scanning the filesystem directly.
    zarr_json = os.path.join(output_dir, "zarr.json")
    if os.path.exists(zarr_json):
        with open(zarr_json) as f:
            meta_obj = json.load(f)
        if "consolidated_metadata" in meta_obj:
            del meta_obj["consolidated_metadata"]
            with open(zarr_json, "w") as f:
                json.dump(meta_obj, f)
            print("[NOTE] Removed stale consolidated metadata from zarr.json")

    print("[FINISH]")


if __name__ == "__main__":
    main()
