#!/usr/bin/env python
# coding: utf-8

# Export images + segmentations to a flat BIDS-like layout for use with
# https://github.com/spinalcordtoolbox/manual-correction
#
# The preprocessing pipeline nests outputs deeply (sub-*/[anat|func]/.../sct_deepseg/...),
# which manual-correction cannot consume directly. This script copies each image and its
# corresponding "_seg" segmentation into:
#
#   <path-data>/derivatives/manual_correction/sub-<ID>/<datatype>/<image>.nii.gz
#   <path-data>/derivatives/manual_correction/derivatives/manual/sub-<ID>/<datatype>/<image>_seg.nii.gz
#
# Usage:
#   python export_manual_correction.py --path-data /path/to/ds007932_260612 [--ids 099 100] [--redo] [--dry-run]

import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--path-data", required=True, help="Path to the dataset root (e.g. .../ds007932_260612)")
parser.add_argument("--ids", nargs="+", default=[""], help="Subject IDs to export (default: all)")
parser.add_argument("--redo", action="store_true", help="Overwrite files that already exist in the output folder")
parser.add_argument("--dry-run", action="store_true", help="Print what would be copied without copying")
args = parser.parse_args()

path_data = os.path.abspath(args.path_data)
preprocessing_dir = os.path.join(path_data, "derivatives", "processing", "preprocessing")
out_root = os.path.join(path_data, "derivatives", "manual_correction")
out_images_dir = out_root
out_labels_dir = os.path.join(out_root, "derivatives", "manual")

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
        parts = rel.split(os.sep)
        sub_id = parts[0].replace("sub-", "")
        if ids != [""] and sub_id not in ids:
            continue
        datatype = parts[1]  # "anat" or "func"

        base = os.path.basename(seg)
        stem = base[: -len("_seg.nii.gz")]
        matches = images_by_stem.get(stem, [])
        if len(matches) != 1:
            print(f"WARNING: skipping {seg} (expected 1 matching image, found {len(matches)})")
            continue

        pairs.append((sub_id, datatype, matches[0], seg))

    return pairs


def copy_file(src, dst, redo, dry_run):
    if os.path.exists(dst) and not redo:
        return
    if dry_run:
        print(f"[dry-run] {src} -> {dst}")
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"{src} -> {dst}")


pairs = find_image_seg_pairs(preprocessing_dir, args.ids)
print(f"Found {len(pairs)} image/segmentation pairs to export.")

for sub_id, datatype, image_src, seg_src in pairs:
    image_dst = os.path.join(out_images_dir, f"sub-{sub_id}", datatype, os.path.basename(image_src))
    seg_dst = os.path.join(out_labels_dir, f"sub-{sub_id}", datatype, os.path.basename(seg_src))

    copy_file(image_src, image_dst, args.redo, args.dry_run)
    copy_file(seg_src, seg_dst, args.redo, args.dry_run)

print("Done.")
