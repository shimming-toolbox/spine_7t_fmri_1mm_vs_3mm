#!/usr/bin/env python
# coding: utf-8

# Quantitative comparison of 1mm vs 3mm fMRI acquisitions.
# Current metric: mean tSNR within the spinal cord mask (native space).

import json, os, glob, argparse
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(path_code, "config", "config_spine_7t_fmri.json")) as f:
    config = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs="+", default=[""])
parser.add_argument("--tasks", nargs="+", default=["motor"])
parser.add_argument("--path-data", required=True)
parser.add_argument("--redo", default="false")
args = parser.parse_args()

path_data = os.path.abspath(args.path_data)
redo = args.redo.lower() == "true"
config["raw_dir"] = path_data
config["code_dir"] = path_code

participants_tsv = pd.read_csv(
    os.path.join(path_code, "config", "participants.tsv"),
    sep="\t", dtype={"participant_id": str}
)
IDs = args.ids if args.ids != [""] else list(participants_tsv["participant_id"])
tasks = args.tasks if args.tasks != [""] else config["design_exp"]["task_names"]

preprocessing_dir = os.path.join(path_data, config["preprocess_dir"]["main_dir"])
tsnr_base_dir = os.path.join(
    path_data,
    config["first_level"]["dir"].format("snr", "").split("sub")[0]
)
out_dir = os.path.join(path_data, "derivatives", "processing", "compare_1mm_3mm")
fig_dir = os.path.join(path_data, "derivatives", "processing", "figures", "compare_1mm_3mm")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# ------------------------------------------------------------------
# 1. Extract mean tSNR within SC mask per subject / acquisition
# ------------------------------------------------------------------
print("=== tSNR within SC mask: extraction ===", flush=True)

csv_path = os.path.join(out_dir, "tsnr_sc_mask.csv")

if not os.path.exists(csv_path) or redo:
    records = []
    for ID in IDs:
        for task in tasks:
            for acq_name in config["design_exp"]["acq_names"]:
                tag = f"task-{task}_acq-{acq_name}"

                # tSNR map (native space) — may include run number
                tsnr_candidates = glob.glob(os.path.join(
                    tsnr_base_dir, f"sub-{ID}", tag,
                    f"sub-{ID}_{tag}*_bold_moco_tsnr.nii.gz"
                ))
                if not tsnr_candidates:
                    print(f"INFO: No tSNR map for sub-{ID} {tag}, skipping.", flush=True)
                    continue
                # use the one with most volumes if multiple runs exist
                tsnr_map = sorted(tsnr_candidates)[-1]

                # SC mask from preprocessing
                seg_path = os.path.join(
                    preprocessing_dir.format(ID), "func", tag,
                    f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz"
                )
                if not os.path.exists(seg_path):
                    print(f"INFO: No SC mask for sub-{ID} {tag}, skipping.", flush=True)
                    continue

                tsnr_data = nib.load(tsnr_map).get_fdata()
                mask_data = nib.load(seg_path).get_fdata()

                vals = tsnr_data[mask_data > 0]
                if vals.size == 0:
                    print(f"WARNING: Empty SC mask for sub-{ID} {tag}, skipping.", flush=True)
                    continue

                records.append({
                    "subject": ID,
                    "task": task,
                    "acq": acq_name,
                    "resolution": "1mm" if "1mm" in acq_name else "3mm",
                    "shim": "shimBase" if "shimBase" in acq_name else "shimSlice",
                    "tsnr_sc": float(np.mean(vals)),
                })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}", flush=True)
else:
    df = pd.read_csv(csv_path)
    print(f"Loaded existing: {csv_path}", flush=True)

print(df.to_string(index=False), flush=True)

# ------------------------------------------------------------------
# 2. Paired figure + Wilcoxon: shimSlice 1mm vs 3mm
# ------------------------------------------------------------------
PAIRS = [
    ("shimSlice+1mm+sms2", "shimSlice+3mm", "shimSlice"),
    ("shimBase+1mm+sms2",  "shimBase+3mm",  "shimBase"),
]
COLORS = {"1mm": "#E64B35", "3mm": "#4DBBD5"}
XLABELS = {
    "shimSlice+1mm+sms2": "1mm (SMS2)",
    "shimSlice+3mm":      "3mm",
    "shimBase+1mm+sms2":  "1mm (SMS2)",
    "shimBase+3mm":       "3mm",
}

for task in tasks:
    df_task = df[df["task"] == task]

    for acq1, acq2, shim_label in PAIRS:
        fig_path = os.path.join(fig_dir, f"tsnr_sc_{shim_label}_{task}_1mm_vs_3mm.png")
        if os.path.exists(fig_path) and not redo:
            print(f"Figure already exists: {fig_path}", flush=True)
            continue

        d1 = df_task[df_task["acq"] == acq1].set_index("subject")["tsnr_sc"]
        d2 = df_task[df_task["acq"] == acq2].set_index("subject")["tsnr_sc"]
        common = d1.index.intersection(d2.index)

        if len(common) < 2:
            print(
                f"WARNING: Only {len(common)} paired subject(s) for "
                f"{acq1} vs {acq2} (task-{task}) — need ≥2, skipping.",
                flush=True,
            )
            continue

        v1 = d1.loc[common].values
        v2 = d2.loc[common].values
        n = len(common)

        stat, pval = stats.wilcoxon(v1, v2)
        if pval < 0.001:
            pstr = "p < 0.001 ***"
        elif pval < 0.01:
            pstr = f"p = {pval:.3f} **"
        elif pval < 0.05:
            pstr = f"p = {pval:.3f} *"
        else:
            pstr = f"p = {pval:.3f} ns"

        # Save stats CSV
        stats_df = pd.DataFrame([{
            "cond1": acq1, "cond2": acq2, "task": task, "N_pairs": n,
            f"mean_{acq1}": v1.mean(), f"std_{acq1}": v1.std(),
            f"mean_{acq2}": v2.mean(), f"std_{acq2}": v2.std(),
            "wilcoxon_stat": stat, "p_value": pval,
            "significance": pstr.split()[-1],
        }])
        stats_path = os.path.join(out_dir, f"tsnr_sc_{shim_label}_{task}_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"Stats ({shim_label}, task-{task}): {pstr}  n={n}", flush=True)

        # --- Figure ---
        fig, ax = plt.subplots(figsize=(3, 4))
        xpos = [0, 1]

        bp = ax.boxplot(
            [v1, v2], positions=xpos, widths=0.45,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="white", linewidth=2),
        )
        for patch, res in zip(bp["boxes"], ["1mm", "3mm"]):
            patch.set_facecolor(COLORS[res])
            patch.set_alpha(0.45)
        for i, element in enumerate(["whiskers", "caps"]):
            for j, line in enumerate(bp[element]):
                line.set_color(COLORS[["1mm", "3mm"][j // 2]])
                line.set_alpha(0.6)

        # Individual subject lines
        for s, (y1, y2) in enumerate(zip(v1, v2)):
            ax.plot(xpos, [y1, y2], "o-", color="dimgray",
                    alpha=0.55, linewidth=1, markersize=4, zorder=3)

        # p-value bracket
        y_max = max(v1.max(), v2.max())
        y_br = y_max * 1.08
        y_txt = y_br * 1.02
        ax.plot([0, 0, 1, 1], [y_max * 1.03, y_br, y_br, y_max * 1.03],
                "k-", linewidth=1)
        ax.text(0.5, y_txt, pstr, ha="center", va="bottom", fontsize=9)

        ax.set_xticks(xpos)
        ax.set_xticklabels([XLABELS[acq1], XLABELS[acq2]], fontsize=10)
        ax.set_ylabel("Mean tSNR within SC mask", fontsize=10)
        ax.set_title(f"tSNR: 1mm vs 3mm\n{shim_label}, task-{task}  (n={n})", fontsize=10)
        ax.set_xlim(-0.6, 1.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}", flush=True)

print("=== compare_workflow done ===", flush=True)
