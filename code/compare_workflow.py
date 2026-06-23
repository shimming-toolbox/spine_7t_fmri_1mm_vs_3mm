#!/usr/bin/env python
# coding: utf-8

# Quantitative comparison of 1mm vs 3mm fMRI acquisitions.
# Current metric: mean tSNR within the spinal cord mask (native space).
# Compares: shimSlice+1mm+sms2 (raw), shimSlice+1mm+sms2+smooth3mm, shimSlice+3mm.

import json, os, glob, argparse, sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(path_code, "code"))
from utils import compute_tsnr_map

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
tsnr_precomp_dir = os.path.join(
    path_data,
    config["first_level"]["dir"].format("snr", "").split("sub")[0]
)
out_dir = os.path.join(path_data, "derivatives", "processing", "compare_1mm_3mm")
fig_dir = os.path.join(path_data, "derivatives", "processing", "figures", "compare_1mm_3mm")
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# Acquisitions to include: regular + smooth3mm derived
REGULAR_ACQS = config["design_exp"]["acq_names"]
DERIVED_ACQS = [k for k in config.get("derived_acq", {}) if "smooth3mm" in k]
ALL_ACQS = REGULAR_ACQS + DERIVED_ACQS


def _get_tsnr_for_derived(ID, task, acq_name, redo):
    """Compute (or load cached) tSNR for a derived acquisition from its moco file."""
    tag = f"task-{task}_acq-{acq_name}"
    moco_candidates = sorted(glob.glob(os.path.join(
        preprocessing_dir.format(ID), "func", tag,
        "sct_fmri_moco", f"sub-{ID}_{tag}*_bold_moco.nii.gz"
    )))
    if not moco_candidates:
        return None
    # pick longest run
    moco_file = max(moco_candidates, key=lambda f: nib.load(f).shape[3])
    cache_dir = os.path.join(out_dir, "tsnr_maps", f"sub-{ID}", tag)
    return compute_tsnr_map(moco_file, cache_dir, redo)


def _get_seg(ID, task, acq_name):
    """Return SC mask path (directly in the tag folder under preprocessing)."""
    tag = f"task-{task}_acq-{acq_name}"
    return os.path.join(
        preprocessing_dir.format(ID), "func", tag,
        f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz"
    )


# ------------------------------------------------------------------
# 1. Extract mean tSNR within SC mask per subject / acquisition
# ------------------------------------------------------------------
print("=== tSNR within SC mask: extraction ===", flush=True)

csv_path = os.path.join(out_dir, "tsnr_sc_mask.csv")

if not os.path.exists(csv_path) or redo:
    records = []
    for ID in IDs:
        for task in tasks:
            for acq_name in ALL_ACQS:
                tag = f"task-{task}_acq-{acq_name}"
                is_derived = acq_name in DERIVED_ACQS

                # --- locate tSNR map ---
                if is_derived:
                    tsnr_map = _get_tsnr_for_derived(ID, task, acq_name, redo)
                else:
                    candidates = glob.glob(os.path.join(
                        tsnr_precomp_dir, f"sub-{ID}", tag,
                        f"sub-{ID}_{tag}*_bold_moco_tsnr.nii.gz"
                    ))
                    tsnr_map = sorted(candidates)[-1] if candidates else None

                if tsnr_map is None or not os.path.exists(tsnr_map):
                    print(f"INFO: No tSNR map for sub-{ID} {tag}, skipping.", flush=True)
                    continue

                # --- locate SC mask ---
                seg_path = _get_seg(ID, task, acq_name)
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
# 2. Helper: Wilcoxon + annotation string
# ------------------------------------------------------------------
def wilcoxon_str(a, b):
    stat, pval = stats.wilcoxon(a, b)
    if pval < 0.001:
        label = "p < 0.001 ***"
    elif pval < 0.01:
        label = f"p = {pval:.3f} **"
    elif pval < 0.05:
        label = f"p = {pval:.3f} *"
    else:
        label = f"p = {pval:.3f} ns"
    return stat, pval, label


def draw_bracket(ax, x1, x2, y_top, label, fontsize=8):
    """Draw a significance bracket between positions x1 and x2 at height y_top."""
    tick = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y_top - tick, y_top, y_top, y_top - tick],
            "k-", linewidth=1)
    ax.text((x1 + x2) / 2, y_top + tick * 0.5, label,
            ha="center", va="bottom", fontsize=fontsize)


# ------------------------------------------------------------------
# 3. Three-condition figure: 1mm | smooth3mm | 3mm  (per shim type)
# ------------------------------------------------------------------
SHIM_TRIPLETS = [
    (
        "shimSlice+1mm+sms2",
        "shimSlice+1mm+sms2+smooth3mm",
        "shimSlice+3mm",
        "shimSlice",
    ),
]

COLORS_3 = ["#E64B35", "#F39B7F", "#4DBBD5"]   # 1mm, smooth3mm, 3mm
XLABELS_3 = {
    "shimSlice+1mm+sms2":           "1mm\n(SMS2)",
    "shimSlice+1mm+sms2+smooth3mm": "1mm\n(smooth3mm)",
    "shimSlice+3mm":                "3mm",
}

for task in tasks:
    df_task = df[df["task"] == task]

    for acq1, acq2, acq3, shim_label in SHIM_TRIPLETS:
        fig_path = os.path.join(fig_dir, f"tsnr_sc_{shim_label}_{task}_triplet.png")
        if os.path.exists(fig_path) and not redo:
            print(f"Figure already exists: {fig_path}", flush=True)
            continue

        d = {
            a: df_task[df_task["acq"] == a].set_index("subject")["tsnr_sc"]
            for a in [acq1, acq2, acq3]
        }

        # subjects present in all three
        common_all = d[acq1].index.intersection(d[acq2].index).intersection(d[acq3].index)
        # subjects present in at least acq2 & acq3 (key comparison)
        common_23 = d[acq2].index.intersection(d[acq3].index)

        if len(common_23) < 2:
            print(
                f"WARNING: Only {len(common_23)} paired subject(s) for "
                f"{acq2} vs {acq3} (task-{task}) — need ≥2, skipping.",
                flush=True,
            )
            continue

        # values for the paired stats
        v2 = d[acq2].loc[common_23].values
        v3 = d[acq3].loc[common_23].values
        stat23, pval23, pstr23 = wilcoxon_str(v2, v3)

        # save stats
        rows = []
        for a, b, va, vb, n in [
            (acq2, acq3, v2, v3, len(common_23)),
        ]:
            s, p, ps = wilcoxon_str(va, vb)
            rows.append({
                "cond1": a, "cond2": b, "task": task, "N_pairs": n,
                f"mean_cond1": va.mean(), f"std_cond1": va.std(),
                f"mean_cond2": vb.mean(), f"std_cond2": vb.std(),
                "wilcoxon_stat": s, "p_value": p, "significance": ps.split()[-1],
            })
        if len(common_all) >= 2:
            v1 = d[acq1].loc[common_all].values
            v2_all = d[acq2].loc[common_all].values
            v3_all = d[acq3].loc[common_all].values
            s12, p12, ps12 = wilcoxon_str(v1, v3_all)
            rows.append({
                "cond1": acq1, "cond2": acq3, "task": task, "N_pairs": len(common_all),
                "mean_cond1": v1.mean(), "std_cond1": v1.std(),
                "mean_cond2": v3_all.mean(), "std_cond2": v3_all.std(),
                "wilcoxon_stat": s12, "p_value": p12, "significance": ps12.split()[-1],
            })
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, f"tsnr_sc_{shim_label}_{task}_stats.csv"), index=False
        )

        # --- Figure ---
        fig, ax = plt.subplots(figsize=(4, 4.5))
        xpos = [0, 1, 2]

        # collect per-condition arrays aligned on common_23 subjects
        # (acq1 may not exist for all of common_23)
        acqs = [acq1, acq2, acq3]
        vals_list = []
        for acq in acqs:
            subs = common_23 if acq != acq1 else common_all
            if len(subs) == 0:
                vals_list.append(np.array([]))
            else:
                vals_list.append(d[acq].reindex(subs).dropna().values)

        # boxplots
        bp_data = [v for v in vals_list if v.size > 0]
        bp_pos  = [xpos[i] for i, v in enumerate(vals_list) if v.size > 0]
        bp_cols = [COLORS_3[i] for i, v in enumerate(vals_list) if v.size > 0]

        bp = ax.boxplot(
            bp_data, positions=bp_pos, widths=0.5,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="white", linewidth=2),
        )
        for patch, col in zip(bp["boxes"], bp_cols):
            patch.set_facecolor(col)
            patch.set_alpha(0.45)
        for i, (wh, ca) in enumerate(zip(bp["whiskers"], bp["caps"])):
            wh.set_color(bp_cols[i // 2])
            ca.set_color(bp_cols[i // 2])
            wh.set_alpha(0.6)
            ca.set_alpha(0.6)

        # individual subject lines — only draw for subjects with all 3
        if len(common_all) >= 1:
            for sub in common_all:
                y_vals = [d[a].get(sub, np.nan) for a in acqs]
                ax.plot(xpos, y_vals, "o-", color="dimgray",
                        alpha=0.5, linewidth=1, markersize=4, zorder=3)

        # for subjects that have only acq2 & acq3, draw dashed line between those
        only_23 = common_23.difference(common_all)
        for sub in only_23:
            ax.plot([1, 2], [d[acq2][sub], d[acq3][sub]], "o--", color="dimgray",
                    alpha=0.4, linewidth=1, markersize=4, zorder=3)

        # p-value bracket (smooth3mm vs 3mm — the key comparison)
        ax.set_ylim(bottom=0)
        y_ceil = max(max(v) for v in bp_data) * 1.25
        ax.set_ylim(top=y_ceil)
        draw_bracket(ax, 1, 2, y_ceil * 0.91, pstr23, fontsize=8)

        ax.set_xticks(xpos)
        ax.set_xticklabels([XLABELS_3[a] for a in acqs], fontsize=9)
        ax.set_ylabel("Mean tSNR within SC mask", fontsize=10)
        ax.set_title(
            f"tSNR: 1mm vs 3mm  ({shim_label}, task-{task})\n"
            f"n={len(common_23)} paired (smooth vs native 3mm)",
            fontsize=9,
        )
        ax.set_xlim(-0.6, 2.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}", flush=True)
        print(f"Stats (smooth3mm vs 3mm, task-{task}): {pstr23}  n={len(common_23)}", flush=True)

# ------------------------------------------------------------------
# 4. BOLD sensitivity: peak z and suprathreshold voxel count
# ------------------------------------------------------------------
print("=== BOLD sensitivity: extraction ===", flush=True)

bold_csv_path = os.path.join(out_dir, "bold_sensitivity.csv")
glm_base_dir = os.path.join(path_data, config["first_level"]["dir"].format("glm", "").split("sub")[0])
pam50_mask_path = os.path.join(path_code, "template", config["PAM50_cord"])
Z_THRESHOLD = 3.1  # FPR < 0.001 one-tailed

if not os.path.exists(bold_csv_path) or redo:
    if not os.path.exists(pam50_mask_path):
        print(f"WARNING: PAM50 cord mask not found at {pam50_mask_path}, skipping BOLD sensitivity.", flush=True)
    else:
        pam50_mask = nib.load(pam50_mask_path).get_fdata() > 0
        bold_records = []

        for ID in IDs:
            for task in tasks:
                for acq_name in ALL_ACQS:
                    tag = f"task-{task}_acq-{acq_name}"
                    # z-map in PAM50 space — contrast: trial_RH-rest
                    zmap_candidates = glob.glob(os.path.join(
                        glm_base_dir, f"sub-{ID}", tag,
                        f"sub-{ID}_{tag}*trial_RH-rest*inTemplate.nii.gz"
                    ))
                    if not zmap_candidates:
                        print(f"INFO: No z-map for sub-{ID} {tag}, skipping.", flush=True)
                        continue
                    zmap_path = sorted(zmap_candidates)[-1]

                    zdata = nib.load(zmap_path).get_fdata()
                    zvals = zdata[pam50_mask]
                    zvals = zvals[np.isfinite(zvals)]
                    if zvals.size == 0:
                        continue

                    bold_records.append({
                        "subject": ID,
                        "task": task,
                        "acq": acq_name,
                        "peak_z": float(np.max(zvals)),
                        "n_active": int(np.sum(zvals > Z_THRESHOLD)),
                    })

        bold_df = pd.DataFrame(bold_records)
        bold_df.to_csv(bold_csv_path, index=False)
        print(f"Saved: {bold_csv_path}", flush=True)
        print(bold_df.to_string(index=False), flush=True)
else:
    bold_df = pd.read_csv(bold_csv_path)
    print(f"Loaded existing: {bold_csv_path}", flush=True)
    print(bold_df.to_string(index=False), flush=True)

# Paired figures for each metric
for metric, ylabel in [("peak_z", "Peak z-score within PAM50 SC mask"),
                        ("n_active", f"Suprathreshold voxels (z > {Z_THRESHOLD})")]:
    for task in tasks:
        bdf_task = bold_df[bold_df["task"] == task] if "task" in bold_df.columns else bold_df

        for acq1, acq2, acq3, shim_label in SHIM_TRIPLETS:
            fig_path = os.path.join(fig_dir, f"{metric}_{shim_label}_{task}_triplet.png")
            if os.path.exists(fig_path) and not redo:
                print(f"Figure already exists: {fig_path}", flush=True)
                continue

            d = {
                a: bdf_task[bdf_task["acq"] == a].set_index("subject")[metric]
                for a in [acq1, acq2, acq3]
            }
            common_all = d[acq1].index.intersection(d[acq2].index).intersection(d[acq3].index)
            common_23  = d[acq2].index.intersection(d[acq3].index)

            if len(common_23) < 2:
                print(
                    f"WARNING: Only {len(common_23)} paired subject(s) for "
                    f"{acq2} vs {acq3} ({metric}, task-{task}) — skipping.",
                    flush=True,
                )
                continue

            v2 = d[acq2].loc[common_23].values
            v3 = d[acq3].loc[common_23].values
            stat23, pval23, pstr23 = wilcoxon_str(v2, v3)

            # stats CSV
            pd.DataFrame([{
                "metric": metric, "cond1": acq2, "cond2": acq3,
                "task": task, "N_pairs": len(common_23),
                "mean_cond1": v2.mean(), "std_cond1": v2.std(),
                "mean_cond2": v3.mean(), "std_cond2": v3.std(),
                "wilcoxon_stat": stat23, "p_value": pval23,
                "significance": pstr23.split()[-1],
            }]).to_csv(
                os.path.join(out_dir, f"{metric}_{shim_label}_{task}_stats.csv"), index=False
            )

            # figure
            fig, ax = plt.subplots(figsize=(4, 4.5))
            xpos = [0, 1, 2]
            acqs = [acq1, acq2, acq3]
            vals_list = []
            for acq in acqs:
                subs = common_23 if acq != acq1 else common_all
                vals_list.append(d[acq].reindex(subs).dropna().values if len(subs) > 0 else np.array([]))

            bp_data = [v for v in vals_list if v.size > 0]
            bp_pos  = [xpos[i] for i, v in enumerate(vals_list) if v.size > 0]
            bp_cols = [COLORS_3[i] for i, v in enumerate(vals_list) if v.size > 0]

            bp = ax.boxplot(bp_data, positions=bp_pos, widths=0.5,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="white", linewidth=2))
            for patch, col in zip(bp["boxes"], bp_cols):
                patch.set_facecolor(col); patch.set_alpha(0.45)
            for wh, ca, col in zip(bp["whiskers"], bp["caps"],
                                   [c for c in bp_cols for _ in range(2)]):
                wh.set_color(col); wh.set_alpha(0.6)
                ca.set_color(col); ca.set_alpha(0.6)

            for sub in common_all:
                y_vals = [d[a].get(sub, np.nan) for a in acqs]
                ax.plot(xpos, y_vals, "o-", color="dimgray",
                        alpha=0.5, linewidth=1, markersize=4, zorder=3)
            for sub in common_23.difference(common_all):
                ax.plot([1, 2], [d[acq2][sub], d[acq3][sub]], "o--",
                        color="dimgray", alpha=0.4, linewidth=1, markersize=4, zorder=3)

            ax.set_ylim(bottom=0)
            y_ceil = max(max(v) for v in bp_data) * 1.25
            ax.set_ylim(top=y_ceil)
            draw_bracket(ax, 1, 2, y_ceil * 0.91, pstr23, fontsize=8)

            ax.set_xticks(xpos)
            ax.set_xticklabels([XLABELS_3[a] for a in acqs], fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(
                f"{metric}: 1mm vs 3mm  ({shim_label}, task-{task})\n"
                f"n={len(common_23)} paired (smooth vs native 3mm)",
                fontsize=9,
            )
            ax.set_xlim(-0.6, 2.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            fig.tight_layout()
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {fig_path}", flush=True)
            print(f"Stats {metric} (smooth3mm vs 3mm, task-{task}): {pstr23}  n={len(common_23)}", flush=True)

print("=== compare_workflow done ===", flush=True)
