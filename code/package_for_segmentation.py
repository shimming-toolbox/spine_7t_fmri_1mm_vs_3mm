#!/usr/bin/env python
# coding: utf-8

# Package the func moco_mean images (no segmentations) into a new folder for a
# student to segment from scratch, preserving the same relative directory
# tree as derivatives/processing/preprocessing/ (just pruned down to the
# moco_mean images).
#
# Usage:
#   python package_for_segmentation.py --path-data /path/to/ds007932_260612 \
#       --output /path/to/ds007932_260612/derivatives/student_package \
#       [--ids 099 100] [--exclude-ids 097 098] [--exclude +avg3mm +smooth3mm]

import argparse
import glob
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--path-data", required=True, help="Path to the dataset root (e.g. .../ds007932_260612)")
parser.add_argument("--output", required=True, help="Folder to copy the packaged images into")
parser.add_argument("--ids", nargs="+", default=[""], help="Subject IDs to include (default: all)")
parser.add_argument("--exclude-ids", nargs="+", default=[], help="Subject IDs to exclude (e.g. test/dummy subjects)")
parser.add_argument("--exclude", nargs="+", default=[], help="Skip images whose path contains any of these strings (e.g. +avg3mm +smooth3mm to skip derived acquisitions)")
parser.add_argument("--redo", action="store_true", help="Overwrite files that already exist in the output folder")
args = parser.parse_args()

path_data = os.path.abspath(args.path_data)
output_dir = os.path.abspath(args.output)
preprocessing_dir = os.path.join(path_data, "derivatives", "processing", "preprocessing")

if not os.path.isdir(preprocessing_dir):
    raise FileNotFoundError(f"Preprocessing directory not found: {preprocessing_dir}")

image_pattern = os.path.join(preprocessing_dir, "sub-*", "func", "**", "sct_fmri_moco", "*_bold_moco_mean.nii.gz")
images = sorted(glob.glob(image_pattern, recursive=True))

n_copied = 0
for image_path in images:
    rel = os.path.relpath(image_path, preprocessing_dir)
    sub_id = rel.split(os.sep)[0].replace("sub-", "")
    if args.ids != [""] and sub_id not in args.ids:
        continue
    if sub_id in args.exclude_ids:
        continue
    if any(x in image_path for x in args.exclude):
        continue

    dst = os.path.join(output_dir, rel)
    if os.path.exists(dst) and not args.redo:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(image_path, dst)
    print(f"{image_path} -> {dst}")
    n_copied += 1

print(f"Copied {n_copied} images to {output_dir}")
