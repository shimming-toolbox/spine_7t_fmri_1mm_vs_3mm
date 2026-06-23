#!/usr/bin/env python
# coding: utf-8

# Quantitative comparison of 1mm vs 3mm fMRI acquisitions.
# Metrics: tSNR within SC mask, BOLD sensitivity (peak z, n_active), MI with T2*w GRE.
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

# ------------------------------------------------------------------
# 5. Mutual Information with T2*w GRE (distortion + dropout proxy)
# ------------------------------------------------------------------
# Mean EPI and T2*w are both registered to PAM50 space during
# preprocessing; MI is computed in that common space within the PAM50
# cord mask.  Higher MI = better EPI-anatomy correspondence.
#
# shimBase acquisitions only exist in the rest task, shimSlice in the
# motor task, so we always loop over ALL config tasks (regardless of
# what --tasks was given) to capture both shim conditions.
#
# 4-condition figure compares shimBase vs shimSlice within each
# resolution (3mm | 1mm), using the task where each condition was
# acquired:
#   shimBase+3mm        ← rest
#   shimSlice+3mm       ← motor
#   shimBase+1mm+sms2   ← rest
#   shimSlice+1mm+sms2+smooth3mm ← motor
# ------------------------------------------------------------------
print("=== MI with T2*w GRE: extraction ===", flush=True)

MI_BINS  = 32
mi_cache = os.path.join(out_dir, "t2star_pam50")
os.makedirs(mi_cache, exist_ok=True)
mi_csv   = os.path.join(out_dir, "mi_t2star.csv")

# Always scan all tasks so shimBase rest data is captured
all_tasks_config = config["design_exp"]["task_names"]


def _compute_mi(x, y, bins=MI_BINS):
    """Histogram-based mutual information between two 1-D arrays."""
    hist2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy  = hist2d / hist2d.sum()
    px   = pxy.sum(axis=1, keepdims=True)
    py   = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py)[mask])))


def _t2star_in_pam50(ID):
    """Warp T2*w GRE to PAM50 space and return the path (cached)."""
    out_path = os.path.join(mi_cache, f"sub-{ID}_T2star_inPAM50.nii.gz")
    if os.path.exists(out_path) and not redo:
        return out_path
    t2star = os.path.join(path_data, f"sub-{ID}", "anat", f"sub-{ID}_T2star.nii.gz")
    warp   = os.path.join(
        preprocessing_dir.format(ID),
        "anat", "sct_register_to_template",
        f"sub-{ID}_from-anat_to-PAM50_mode-image_xfm.nii.gz"
    )
    template = os.path.join(path_code, "template", config["PAM50_t2"])
    if not os.path.exists(t2star) or not os.path.exists(warp):
        return None
    cmd = (f"sct_apply_transfo -i {t2star} -d {template} -w {warp} "
           f"-o {out_path} -x spline")
    ret = os.system(cmd)
    return out_path if ret == 0 and os.path.exists(out_path) else None


if not os.path.exists(mi_csv) or redo:
    pam50_mask_data = nib.load(pam50_mask_path).get_fdata() > 0
    mi_records = []

    for ID in IDs:
        t2star_pam50 = _t2star_in_pam50(ID)
        if t2star_pam50 is None:
            print(f"WARNING: Could not warp T2*w for sub-{ID}, skipping.", flush=True)
            continue
        t2_vals = nib.load(t2star_pam50).get_fdata()[pam50_mask_data]

        for task in all_tasks_config:
            for acq_name in ALL_ACQS:
                tag = f"task-{task}_acq-{acq_name}"
                epi_candidates = glob.glob(os.path.join(
                    preprocessing_dir.format(ID), "func", tag,
                    "sct_register_multimodal",
                    f"sub-{ID}_{tag}*_bold_moco_mean_coreg_in_PAM50.nii.gz"
                ))
                if not epi_candidates:
                    print(f"INFO: No mean EPI in PAM50 for sub-{ID} {tag}, skipping.", flush=True)
                    continue
                epi_pam50 = sorted(epi_candidates)[-1]

                epi_vals = nib.load(epi_pam50).get_fdata()[pam50_mask_data]
                valid = np.isfinite(epi_vals) & np.isfinite(t2_vals) & (epi_vals > 0)
                if valid.sum() < 10:
                    continue

                mi_val = _compute_mi(epi_vals[valid], t2_vals[valid])
                mi_records.append({
                    "subject": ID, "task": task, "acq": acq_name, "mi": mi_val
                })
                print(f"sub-{ID} {tag}: MI = {mi_val:.4f}", flush=True)

    mi_df = pd.DataFrame(mi_records)
    mi_df.to_csv(mi_csv, index=False)
    print(f"Saved: {mi_csv}", flush=True)
else:
    mi_df = pd.read_csv(mi_csv)
    print(f"Loaded existing: {mi_csv}", flush=True)

print(mi_df.to_string(index=False), flush=True)

# ------------------------------------------------------------------
# 4-condition figure: shimBase vs shimSlice, within 3mm and 1mm
# ------------------------------------------------------------------
# Each condition is drawn from the task where it was acquired:
#   shimBase+3mm       ← rest   shimSlice+3mm             ← motor
#   shimBase+1mm+sms2  ← rest   shimSlice+1mm+sms2+smooth3mm ← motor
#
# xpos grouped by resolution: [0, 1] = 3mm group, [2.5, 3.5] = 1mm group
# Stats: shimBase vs shimSlice within each group (Wilcoxon signed-rank)
# ------------------------------------------------------------------
MI_4COND = [
    # (acq_name,                      task,    x,    color,     label)
    ("shimBase+3mm",                   "rest",  0,    "#2166AC", "shimBase\n3mm"),
    ("shimSlice+3mm",                  "motor", 1,    "#74ADD1", "shimSlice\n3mm"),
    ("shimBase+1mm+sms2",              "rest",  2.5,  "#D73027", "shimBase\n1mm"),
    ("shimSlice+1mm+sms2+smooth3mm",   "motor", 3.5,  "#F4A582", "shimSlice\n1mm\n(smooth3mm)"),
]

fig_path_4 = os.path.join(fig_dir, "mi_t2star_4cond.png")
if not os.path.exists(fig_path_4) or redo:
    # Build per-condition subject→MI lookup
    mi_ser = {}
    for acq_name, task, *_ in MI_4COND:
        sub_df = mi_df[(mi_df["task"] == task) & (mi_df["acq"] == acq_name)]
        mi_ser[acq_name] = sub_df.set_index("subject")["mi"]

    # Pairs for statistics
    base3_acq, slice3_acq  = MI_4COND[0][0], MI_4COND[1][0]
    base1_acq, slice1_acq  = MI_4COND[2][0], MI_4COND[3][0]
    common_3mm = mi_ser[base3_acq].index.intersection(mi_ser[slice3_acq].index)
    common_1mm = mi_ser[base1_acq].index.intersection(mi_ser[slice1_acq].index)

    stats_rows = []
    for a, b, common, grp in [
        (base3_acq, slice3_acq, common_3mm, "3mm"),
        (base1_acq, slice1_acq, common_1mm, "1mm"),
    ]:
        if len(common) >= 2:
            va = mi_ser[a].loc[common].values
            vb = mi_ser[b].loc[common].values
            s, p, ps = wilcoxon_str(va, vb)
            stats_rows.append({
                "metric": "mi", "group": grp, "cond1": a, "cond2": b,
                "N_pairs": len(common),
                "mean_cond1": va.mean(), "std_cond1": va.std(),
                "mean_cond2": vb.mean(), "std_cond2": vb.std(),
                "wilcoxon_stat": s, "p_value": p, "significance": ps.split()[-1],
            })
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(
            os.path.join(out_dir, "mi_t2star_4cond_stats.csv"), index=False
        )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    xpos_list  = [c[2] for c in MI_4COND]
    color_list = [c[3] for c in MI_4COND]
    label_list = [c[4] for c in MI_4COND]
    vals_all   = [mi_ser[c[0]].dropna().values for c in MI_4COND]

    bp = ax.boxplot(vals_all, positions=xpos_list, widths=0.45,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], color_list):
        patch.set_facecolor(col); patch.set_alpha(0.5)
    for i, (wh, ca) in enumerate(zip(bp["whiskers"], bp["caps"])):
        wh.set_color(color_list[i // 2]); wh.set_alpha(0.6)
        ca.set_color(color_list[i // 2]); ca.set_alpha(0.6)

    # Individual lines within each resolution group
    for sub in common_3mm:
        ax.plot([0, 1], [mi_ser[base3_acq][sub], mi_ser[slice3_acq][sub]],
                "o-", color="dimgray", alpha=0.5, linewidth=1, markersize=4, zorder=3)
    for sub in common_1mm:
        ax.plot([2.5, 3.5], [mi_ser[base1_acq][sub], mi_ser[slice1_acq][sub]],
                "o-", color="dimgray", alpha=0.5, linewidth=1, markersize=4, zorder=3)

    ax.set_ylim(bottom=0)
    y_max = max(v.max() for v in vals_all if v.size > 0) * 1.35
    ax.set_ylim(top=y_max)

    bracket_y3 = y_max * 0.86
    bracket_y1 = y_max * 0.86
    for (a, b, common, grp), (bx1, bx2), by in [
        ((base3_acq, slice3_acq, common_3mm, "3mm"), (0, 1),     bracket_y3),
        ((base1_acq, slice1_acq, common_1mm, "1mm"), (2.5, 3.5), bracket_y1),
    ]:
        if len(common) >= 2:
            va = mi_ser[a].loc[common].values
            vb = mi_ser[b].loc[common].values
            _, _, ps = wilcoxon_str(va, vb)
            draw_bracket(ax, bx1, bx2, by, ps, fontsize=8)
            print(f"Stats MI {grp} (shimBase vs shimSlice): {ps}  n={len(common)}", flush=True)

    # Vertical separator between resolution groups
    ax.axvline(x=1.75, color="lightgray", linewidth=0.8, linestyle="--", zorder=0)
    ax.text(0.5,  y_max * 0.97, "3mm",                ha="center", va="top", fontsize=9, color="gray")
    ax.text(3.0,  y_max * 0.97, "1mm (smooth3mm)",    ha="center", va="top", fontsize=9, color="gray")

    ax.set_xticks(xpos_list)
    ax.set_xticklabels(label_list, fontsize=8.5)
    ax.set_ylabel("Mutual Information with T2*w GRE\n(PAM50 space, within SC mask)", fontsize=9)
    ax.set_title("MI with T2*w: shimBase vs shimSlice\n(3mm and 1mm acquisitions)", fontsize=9)
    ax.set_xlim(-0.6, 4.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(fig_path_4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fig_path_4}", flush=True)
else:
    print(f"Figure already exists: {fig_path_4}", flush=True)

print("=== compare_workflow done ===", flush=True)
