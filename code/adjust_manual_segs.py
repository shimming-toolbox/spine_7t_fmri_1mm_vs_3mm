"""
Adjust the manual EPI segmentations, drawn on the moco means of a previous preprocessing run, to the
moco means of a new run (#125). Needed when motion correction changes, e.g. the moco-dl model (#124).

For each acquisition with a manual SC segmentation (`*_bold_moco_mean_label-SC_seg.nii.gz`):
  1. Mask: 30 mm cylinder around the manual SC segmentation, on the new mean.
  2. Register old mean -> new mean, slicewise rigid (sct_register_multimodal), with QC.
     Not algo=translation,slicewise=1: SCT inverts the sign of its per-slice translations
     (registration/algorithms.py, generate_warping_field), so it moves each slice the wrong way.
  3. Apply the warp to the SC and CSF segmentations (linear interpolation, binarized at 0.5).
  4. QC of the SC segmentation on the new mean, before and after adjustment, and fit scores against
     `sct_deepseg sc_epi` on the old mean (original segmentation) and on the new mean (original and
     adjusted segmentations).

Nothing is written to the manual folder: adjusted segmentations go to <output>/<acquisition>/.

Usage:
    python adjust_manual_segs.py --manual derivatives/manual \
        --old derivatives/processing_<old>/preprocessing --new derivatives/processing_<new>/preprocessing \
        --output <dir>
"""
import argparse
import glob
import os
import subprocess

import nibabel as nib
import numpy as np
import pandas as pd

PARAM = "step=1,type=im,algo=rigid,metric=CC,slicewise=1"


def run(cmd):
    subprocess.run(cmd, check=True, capture_output=True)


def find_mean(root, stem):
    f = glob.glob(os.path.join(root, stem.split("_")[0], "func", "*", "sct_fmri_moco", f"{stem}_moco_mean.nii.gz"))
    return f[0] if f else None


def fit(seg_f, ref_f):
    """Dice and per-slice centroid distance (mm) between a segmentation and a reference segmentation."""
    s = nib.load(seg_f).get_fdata() > 0.5
    ref = nib.load(ref_f)
    r = ref.get_fdata() > 0.5
    zooms = np.array(ref.header.get_zooms()[:2])
    d = np.array([np.linalg.norm((np.array(np.nonzero(s[..., z])).mean(1) - np.array(np.nonzero(r[..., z])).mean(1)) * zooms)
                  for z in range(s.shape[2]) if s[..., z].any() and r[..., z].any()])
    return 2 * (s & r).sum() / (s.sum() + r.sum()), d.mean(), d.max(), int((d > zooms[0]).sum())


def deepseg(mean_f, out_f):
    if not os.path.exists(out_f):
        run(["sct_deepseg", "sc_epi", "-i", mean_f, "-o", out_f, "-v", "0"])
    return out_f


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manual", required=True, help="Manual segmentations folder (derivatives/manual)")
    parser.add_argument("--old", required=True, help="Preprocessing folder of the run the segmentations were drawn on")
    parser.add_argument("--new", required=True, help="Preprocessing folder of the new run")
    parser.add_argument("--output", required=True, help="Output folder")
    args = parser.parse_args()
    qc = os.path.join(args.output, "qc")

    rows = []
    for sc_f in sorted(glob.glob(os.path.join(args.manual, "sub-*", "func", "*_bold_moco_mean_label-SC_seg.nii.gz"))):
        stem = os.path.basename(sc_f).replace("_moco_mean_label-SC_seg.nii.gz", "")
        sub = stem.split("_")[0]
        contrast = stem.split("_", 1)[1].replace("_bold", "")
        old_f, new_f = find_mean(args.old, stem), find_mean(args.new, stem)
        if old_f is None or new_f is None:
            print(f"skip {stem}: no {'old' if old_f is None else 'new'} mean")
            continue
        d = os.path.join(args.output, stem)
        os.makedirs(d, exist_ok=True)

        # 1. mask around the cord, on the new mean
        mask_f = os.path.join(d, "mask.nii.gz")
        run(["sct_create_mask", "-i", new_f, "-p", f"centerline,{sc_f}", "-size", "30mm", "-o", mask_f, "-v", "0"])

        # 2. old mean -> new mean, slicewise rigid. QC needs a destination seg: deepseg on the new mean.
        new_auto = deepseg(new_f, os.path.join(d, "new_mean_deepseg.nii.gz"))
        run(["sct_register_multimodal", "-i", old_f, "-d", new_f, "-m", mask_f, "-param", PARAM,
             "-ofolder", d, "-owarp", os.path.join(d, "warp_old2new.nii.gz"),
             "-owarpinv", os.path.join(d, "warp_new2old.nii.gz"),
             "-qc", qc, "-qc-dataset", "seg-adjust", "-qc-subject", sub,
             "-qc-contrast", f"{contrast} | registration old->new", "-dseg", new_auto, "-v", "0"])

        # 3. move the manual segmentations (SC, and CSF when it exists)
        adjusted = {}
        for label in ["SC", "CSF"]:
            seg_f = sc_f.replace("label-SC", f"label-{label}")
            if not os.path.exists(seg_f):
                continue
            out_f = os.path.join(d, os.path.basename(seg_f).replace("_seg.nii.gz", "_seg_adjusted.nii.gz"))
            run(["sct_apply_transfo", "-i", seg_f, "-d", new_f, "-w", os.path.join(d, "warp_old2new.nii.gz"),
                 "-x", "linear", "-o", out_f, "-v", "0"])
            run(["sct_maths", "-i", out_f, "-bin", "0.5", "-o", out_f, "-v", "0"])
            adjusted[label] = out_f

        # 4. QC and fit scores
        for tag, seg in (("before", sc_f), ("after", adjusted["SC"])):
            run(["sct_qc", "-i", new_f, "-s", seg, "-p", "sct_deepseg_sc", "-qc", qc, "-qc-dataset", "seg-adjust",
                 "-qc-subject", sub, "-qc-contrast", f"{contrast} | SC seg on new mean, {tag}", "-v", "0"])
        old_auto = deepseg(old_f, os.path.join(d, "old_mean_deepseg.nii.gz"))
        row = {"seg": stem, "has_csf": "CSF" in adjusted}
        for k, (seg, ref) in {"old": (sc_f, old_auto), "new": (sc_f, new_auto), "adjusted": (adjusted["SC"], new_auto)}.items():
            row[f"dice_{k}"], row[f"offset_mean_{k}"], row[f"offset_max_{k}"], row[f"slices_off_1vox_{k}"] = fit(seg, ref)
        rows.append(row)
        print(f"done {stem}: dice old {row['dice_old']:.2f} / new {row['dice_new']:.2f} / adjusted {row['dice_adjusted']:.2f}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(args.output, "seg_adjust.csv"), index=False)
    print(f"\nQC: {qc}/index.html")


if __name__ == "__main__":
    main()
