#!/usr/bin/env python
# coding: utf-8

# Generate a slicer-cart cohort.csv with one row per image/segmentation pair,
# instead of one row per subject. slicer-cart's cohort model has no concept of
# "resource groups" within a case (see neuropoly/slicer-cart#200), so reviewing
# the ~6 segmentations/subject in this project one at a time (rather than all
# at once in a single case) requires giving each pair its own row/uid.
#
# Unlike export_manual_correction.py, this does NOT copy any files: the cohort
# CSV just references the existing files in place, with paths relative to
# --path-data.
#
# Usage:
#   python generate_slicercart_cohort.py --path-data /path/to/ds007932_260612 \
#       --output /path/to/ds007932_260612/cohort.csv [--ids 099 100]

import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path-data", required=True, help="Path to the dataset root (e.g. .../ds007932_260612)")
parser.add_argument("--output", required=True, help="Path to the cohort CSV to create")
parser.add_argument("--ids", nargs="+", default=[""], help="Subject IDs to include (default: all)")
args = parser.parse_args()

path_data = os.path.abspath(args.path_data)
preprocessing_dir = os.path.join(path_data, "derivatives", "processing", "preprocessing")

if not os.path.isdir(preprocessing_dir):
    raise FileNotFoundError(f"Preprocessing directory not found: {preprocessing_dir}")


def find_image_seg_pairs(preprocessing_dir, ids):
    """Pair each sct_deepseg/*_seg.nii.gz with its source image (same basename, minus '_seg')."""
    seg_pattern = os.path.join(preprocessing_dir, "sub-*", "**", "sct_deepseg", "*_seg.nii.gz")
    segs = sorted(glob.glob(seg_pattern, recursive=True))

    # Index all candidate images (any .nii.gz that isn't itself a "_seg" file) by basename stem.
    image_pattern = os.path.join(preprocessing_dir, "sub-*", "**", "*.nii.gz")
    images_by_stem = {}
    for f in glob.glob(image_pattern, recursive=True):
        base = os.path.basename(f)
        if base.endswith("_seg.nii.gz"):
            continue
        stem = base[: -len(".nii.gz")]
        images_by_stem.setdefault(stem, []).append(f)

    pairs = []
    for seg in segs:
        rel = os.path.relpath(seg, preprocessing_dir)
        sub_id = rel.split(os.sep)[0].replace("sub-", "")
        if ids != [""] and sub_id not in ids:
            continue

        base = os.path.basename(seg)
        stem = base[: -len("_seg.nii.gz")]
        matches = images_by_stem.get(stem, [])
        if len(matches) != 1:
            print(f"WARNING: skipping {seg} (expected 1 matching image, found {len(matches)})")
            continue

        pairs.append((stem, matches[0], seg))

    return pairs


pairs = find_image_seg_pairs(preprocessing_dir, args.ids)
pairs.sort(key=lambda p: p[0])
print(f"Found {len(pairs)} image/segmentation pairs.")

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
with open(args.output, "w") as fp:
    fp.write("uid,image_volume_reference,seg_segmentation_editable\n")
    for uid, image_path, seg_path in pairs:
        image_rel = os.path.relpath(image_path, path_data)
        seg_rel = os.path.relpath(seg_path, path_data)
        fp.write(f"{uid},{image_rel},{seg_rel}\n")

print(f"Wrote cohort file: {args.output}")
print(f"Configure slicer-cart's data path to: {path_data}")
