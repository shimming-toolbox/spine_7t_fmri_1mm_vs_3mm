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
# Also writes the <output>.json sidecar slicer-cart expects next to the CSV
# (same basename, ".json" extension). Without it, CohortModel silently resets
# its case/resource maps on load and the task cannot start
# (see neuropoly/slicer-cart#201) — so the sidecar isn't optional.
#
# Usage:
#   python generate_slicercart_cohort.py --path-data /path/to/ds007932_260612 \
#       --output /path/to/ds007932_260612/cohort.csv [--ids 099 100] [--exclude task-motor] [--no-seg]

import argparse
import glob
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--path-data", required=True, help="Path to the dataset root (e.g. .../ds007932_260612)")
parser.add_argument("--output", required=True, help="Path to the cohort CSV to create")
parser.add_argument("--ids", nargs="+", default=[""], help="Subject IDs to include (default: all)")
parser.add_argument("--exclude", nargs="+", default=[], help="Skip pairs whose image path contains any of these strings (e.g. task-motor)")
parser.add_argument("--no-seg", action="store_true", help="Leave the segmentation column blank even if a sct_deepseg output exists, so slicer-cart starts from an empty mask")
args = parser.parse_args()

path_data = os.path.abspath(args.path_data)
preprocessing_dir = os.path.join(path_data, "derivatives", "processing", "preprocessing")

if not os.path.isdir(preprocessing_dir):
    raise FileNotFoundError(f"Preprocessing directory not found: {preprocessing_dir}")


def case_root(preprocessing_dir, path):
    """The sub-<ID>[/ses-<ID>] prefix of a path, used to scope image/seg matching to one case."""
    rel = os.path.relpath(path, preprocessing_dir)
    parts = rel.split(os.sep)
    if len(parts) > 1 and parts[1].startswith("ses-"):
        return os.path.join(parts[0], parts[1])
    return parts[0]


def find_image_seg_pairs(preprocessing_dir, ids, exclude):
    """Pair each sct_deepseg/*_seg.nii.gz with its source image (same basename, minus '_seg')."""
    seg_pattern = os.path.join(preprocessing_dir, "sub-*", "**", "sct_deepseg", "*_seg.nii.gz")
    segs = sorted(glob.glob(seg_pattern, recursive=True))
    segs = [s for s in segs if not any(x in s for x in exclude)]

    # Index all candidate images (any .nii.gz that isn't itself a "_seg" file) by
    # (case root, basename stem), so identical basenames in different subjects/
    # sessions (e.g. copy-pasted test data) can't be confused with each other.
    image_pattern = os.path.join(preprocessing_dir, "sub-*", "**", "*.nii.gz")
    images_by_key = {}
    for f in glob.glob(image_pattern, recursive=True):
        base = os.path.basename(f)
        if base.endswith("_seg.nii.gz"):
            continue
        stem = base[: -len(".nii.gz")]
        key = (case_root(preprocessing_dir, f), stem)
        images_by_key.setdefault(key, []).append(f)

    pairs = []
    seen_uids = set()
    for seg in segs:
        rel = os.path.relpath(seg, preprocessing_dir)
        sub_id = rel.split(os.sep)[0].replace("sub-", "")
        if ids != [""] and sub_id not in ids:
            continue

        base = os.path.basename(seg)
        stem = base[: -len("_seg.nii.gz")]
        root = case_root(preprocessing_dir, seg)
        matches = images_by_key.get((root, stem), [])
        if len(matches) != 1:
            print(f"WARNING: skipping {seg} (expected 1 matching image, found {len(matches)})")
            continue

        # The stem is normally already unique (it encodes subject + acquisition),
        # but disambiguate just in case two cases share one (e.g. mismatched
        # filenames from copy-pasted test data).
        uid = stem
        if uid in seen_uids:
            uid = f"{root.replace(os.sep, '_')}_{stem}"
        seen_uids.add(uid)

        pairs.append((uid, matches[0], seg))

    return pairs


pairs = find_image_seg_pairs(preprocessing_dir, args.ids, args.exclude)
pairs.sort(key=lambda p: p[0])
print(f"Found {len(pairs)} image/segmentation pairs.")

os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
case_paths = {}
with open(args.output, "w") as fp:
    fp.write("uid,image_volume_reference,seg_segmentation_editable\n")
    for uid, image_path, seg_path in pairs:
        image_rel = os.path.relpath(image_path, path_data)
        seg_rel = "" if args.no_seg else os.path.relpath(seg_path, path_data)
        fp.write(f"{uid},{image_rel},{seg_rel}\n")

        # Search dir for this case: the shared parent of the image and seg
        # (e.g. the func/task-.../ or anat/ folder), so slicer-cart's
        # resource-editing dialogs can find both. Falls back to the image's
        # own folder when there's no seg to share a parent with (--no-seg).
        image_dir = os.path.dirname(image_rel)
        if seg_rel:
            search_dir = os.path.commonpath([image_dir, os.path.dirname(seg_rel)])
        else:
            search_dir = image_dir
        case_paths[uid] = [search_dir]

sidecar_path = os.path.splitext(args.output)[0] + ".json"
sidecar_data = {
    "cohort_version": "0.2.0",
    "case_paths": case_paths,
    "filters": {
        "image_volume_reference": {
            "original_name": "image",
            "resource_type": "volume_reference",
            "include": [],
            "exclude": ["_seg"],
            "extension": ".nii.gz",
        },
        "seg_segmentation_editable": {
            "original_name": "seg",
            "resource_type": "segmentation_editable",
            "include": ["_seg"],
            "exclude": [],
            "extension": ".nii.gz",
        },
    },
}
with open(sidecar_path, "w") as fp:
    json.dump(sidecar_data, fp, indent=2)

print(f"Wrote cohort file: {args.output}")
print(f"Wrote sidecar file: {sidecar_path}")
print(f"Configure slicer-cart's data path to: {path_data}")
