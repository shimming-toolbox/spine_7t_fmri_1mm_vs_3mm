#!/usr/bin/env python
# coding: utf-8

# Generate all publication figures from pre-computed results.
# Run after firstlevel_workflow, secondlevel_workflow, and compare_workflow have completed.

import re, json, sys, os, glob, argparse
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
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--redo", default="False")
parser.add_argument("--path-data", required=True)
args = parser.parse_args()

IDs = args.ids
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)

config["raw_dir"] = path_data
config["code_dir"] = path_code

participants_tsv = pd.read_csv(
    os.path.join(path_code, "config", "participants.tsv"),
    sep='\t', dtype={'participant_id': str}
)
if IDs == [""]:
    IDs = list(participants_tsv["participant_id"])

sys.path.append(os.path.join(path_code, "code"))
import postprocess
from preprocess import manual_label_filename
import figures as figures_module

figures = figures_module.Figures_main(config, IDs=IDs)

# Directories
first_level_dir  = os.path.join(config["raw_dir"], config["first_level"]["dir"])
second_level_dir = os.path.join(config["raw_dir"], config["second_level"]["dir"])
output_fig_second_level  = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "second_level")
output_fig_first_level   = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "first_level")
output_fig_preprocessing = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "preprocessing")
output_fig = output_fig_second_level  # kept for GLM/ICC figures
os.makedirs(output_fig_second_level, exist_ok=True)
os.makedirs(output_fig_first_level, exist_ok=True)
os.makedirs(output_fig_preprocessing, exist_ok=True)

print("=== figures_workflow Start ===", flush=True)
print(f"Participants: {IDs}", flush=True)
print("===================================", flush=True)

# ------------------------------------------------------------------
# 1. EPI comparison figure
# ------------------------------------------------------------------
print("", flush=True)
print("=== EPI comparison figure ===", flush=True)
try:
    epi_fig = postprocess.EpiComparison(config, IDs, redo)
    epi_fig.create_figure()
except Exception as e:
    print(f"WARNING: EPI comparison figure skipped: {e}", flush=True)

# ------------------------------------------------------------------
# 2. First-level maps (shimBase vs. shimSlice per participant)
# ------------------------------------------------------------------
print("", flush=True)
print("=== First-level maps ===", flush=True)
fig_dir_first = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "first_level")
os.makedirs(fig_dir_first, exist_ok=True)
common_mask_fname = os.path.join(first_level_dir.format('glm', "").split("sub")[0], "common_mask_PAM50.nii.gz")

# Compare native 3mm vs. 1mm(smooth3mm) for a single shim condition (shimSlice), rather
# than shimBase vs shimSlice at 3mm: shimBase MOTOR 1mm data only exists for sub-099, so
# it can't support a consistent group comparison, while shimSlice has 3mm and 1mm
# (smooth3mm) GLM results for every subject -- and 3mm vs 1mm is this project's main axis.
first_level_acqs = ["shimSlice+3mm", "shimSlice+1mm+sms2+smooth3mm"]

i_fnames_by_runs = []
used_ids = []
for ID in IDs:
    i_fnames_runs = []
    for task_name in config["design_exp"]["task_names"]:
        for acq_name in first_level_acqs:
            tag = "task-" + task_name + "_acq-" + acq_name
            # Glob the GLM result directly (rather than checking for a matching raw BOLD
            # file first): derived acquisitions like +smooth3mm have no raw BOLD of their
            # own. Sorted so that, when a subject has multiple runs (e.g. sub-099), only
            # the first (run-01) is used -- every subject then contributes the same 2 columns.
            matches = sorted(glob.glob(os.path.join(
                first_level_dir.format("glm", ID), tag,
                f"*{tag}*trial_RH-rest*inTemplate.nii.gz"
            )))
            if matches:
                i_fnames_runs.append(matches[0])
    if i_fnames_runs:
        i_fnames_by_runs.append(i_fnames_runs)
        used_ids.append(ID)

try:
    figures.plot_first_level_maps(
        i_fnames=i_fnames_by_runs,
        output_fname=os.path.join(fig_dir_first, f"first_level_task_by_runs_n{len(i_fnames_by_runs)}.png"),
        background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
        mask_fname=common_mask_fname,
        titles=["3mm", "1mm (smooth3mm)"],
        task_name=tag,
        participant_ids=used_ids,
        verbose=True,
        redo=redo)
except Exception as e:
    print(f"WARNING: First-level maps skipped: {e}", flush=True)

# ------------------------------------------------------------------
# 3. tSNR / sSNR 4-condition plots  →  see section 7g (generated after metrics)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# 4. tSNR group maps (native 3mm vs avg3mm)
# ------------------------------------------------------------------
print("", flush=True)
print("=== tSNR group maps ===", flush=True)
avg_acq_names = {k: v for k, v in config.get("derived_acq", {}).items() if "n_slices_avg" in v}
try:
    for shim_label in ["shimBase", "shimSlice"]:
        acq_3mm    = next((a for a in config["design_exp"]["acq_names"] if shim_label in a and "3mm" in a), None)
        acq_avg3mm = next((a for a in avg_acq_names if shim_label in a), None)
        if acq_3mm is None or acq_avg3mm is None:
            print(f"INFO: Could not find 3mm or avg3mm acq for {shim_label}, skipping.", flush=True)
            continue
        f_3mm    = glob.glob(os.path.join(second_level_dir.format("snr"), f"tsnr_n*_{acq_3mm}_avg_in_PAM50.nii.gz"))
        f_avg3mm = glob.glob(os.path.join(second_level_dir.format("snr"), f"tsnr_n*_{acq_avg3mm}_avg_in_PAM50.nii.gz"))
        if not f_3mm or not f_avg3mm:
            print(f"INFO: Missing tSNR avg map for {shim_label}, skipping.", flush=True)
            continue
        figures.plot_fmri_maps(
            i_fnames=[f_3mm[0], f_avg3mm[0]],
            output_fname=os.path.join(output_fig, f"n{len(IDs)}_tsnr_{shim_label}_3mm_vs_avg3mm_map.png"),
            stat_min=5, stat_max=16, cmap='turbo', cbar_label='tSNR',
            titles=[f"{shim_label} 3mm", f"{shim_label} avg3mm"],
            background_fname=os.path.join(path_code, "template", config["PAM50_t2"]), redo=redo)
except Exception as e:
    print(f"WARNING: tSNR group maps failed: {e}", flush=True)

# ------------------------------------------------------------------
# 5. GLM group maps + bar + dist + combined
# ------------------------------------------------------------------
print("", flush=True)
print("=== GLM group maps ===", flush=True)
try:
    glm_plot = {}; glm_axial_plot = {}; bar_plot = {}; dist_plot = {}
    for cluster_corr in [0.01, 0.001]:
        glm_plot[cluster_corr] = {}; glm_axial_plot[cluster_corr] = {}
        bar_plot[cluster_corr] = {}; dist_plot[cluster_corr] = {}
        z_slices = [280, 266, 256, 243, 225]
        for vox_thr in [0.005]:
            i_fnames_glm = []
            metrics_csvs = []
            values_csvs  = []
            glm_dir = os.path.join(
                second_level_dir.format("glm"),
                f"cluster_p{cluster_corr}_vox{vox_thr}_perm10000"
            )
            for acq_name in config["design_exp"]["acq_names"]:
                tag = "task-motor_acq-" + acq_name
                nii_cands = glob.glob(os.path.join(glm_dir, tag, f"*_{tag}_t_clustercorrected.nii.gz"))
                if not nii_cands:
                    continue
                nii_path = nii_cands[0]
                i_fnames_glm.append(nii_path)
                base = nii_path.split(".nii.gz")[0]
                if os.path.exists(base + "_metrics.csv"):
                    metrics_csvs.append(base + "_metrics.csv")
                if os.path.exists(base + "_values.csv"):
                    values_csvs.append(base + "_values.csv")

            if not i_fnames_glm:
                print(f"INFO: No GLM maps for cluster={cluster_corr} vox={vox_thr}, skipping.", flush=True)
                continue

            glm_plot[cluster_corr][vox_thr] = figures.plot_fmri_maps(
                i_fnames=i_fnames_glm,
                output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_{cluster_corr}_vox{vox_thr}_avg_map.png"),
                stat_min=3, stat_max=6, cbar_label='t-value', z_slices=z_slices,
                background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]), redo=redo)

            glm_axial_plot[cluster_corr][vox_thr] = figures.plot_fmri_maps_axial(
                i_fnames=i_fnames_glm,
                output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_axial_{cluster_corr}_vox{vox_thr}_avg_map.png"),
                stat_min=3, stat_max=6, cbar_label='t-value',
                background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
                underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]),
                z_slices=z_slices, n_slices=len(z_slices), redo=redo)

            if metrics_csvs:
                bar_plot[cluster_corr][vox_thr] = figures.bar_plot(
                    csv_pair=metrics_csvs, figsize=(2, 2.7),
                    output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_cluster{cluster_corr}_vox{vox_thr}_nb_vox.png"),
                    redo=redo)

            if len(values_csvs) >= 2:
                dist_plot[cluster_corr][vox_thr] = figures.plot_dist(
                    csv_pair=[values_csvs[1], values_csvs[0]],
                    maps_name=["shimSlice", "shimBase"],
                    colors=["#ED263F", "#ADA8A8"], figsize=(2, 2.7),
                    output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_cluster{cluster_corr}_vox{vox_thr}_distr.png"),
                    redo=redo)

            if vox_thr in glm_plot.get(cluster_corr, {}):
                figures.combine_plots(
                    output_fname=os.path.join(output_fig, f"n{len(IDs)}_combined_task_cluster_{cluster_corr}_vox_{vox_thr}_plots.png"),
                    map_files=[glm_plot[cluster_corr][vox_thr]],
                    axial_files=[glm_axial_plot[cluster_corr][vox_thr]],
                    graph_files=[
                        bar_plot[cluster_corr].get(vox_thr),
                        dist_plot.get(cluster_corr, {}).get(vox_thr)
                    ],
                    label_idx=True, figsize=(5, 3.8), redo=True)
except Exception as e:
    print(f"WARNING: GLM group maps failed: {e}", flush=True)

# ------------------------------------------------------------------
# 5b. GLM uncorrected maps (vox p<0.05, for exploratory display with small N)
# ------------------------------------------------------------------
print("", flush=True)
print("=== GLM uncorrected group maps (vox p<0.05) ===", flush=True)
try:
    vox_thr_unc = 0.05
    # Use glob to find whichever perm* directory was computed (e.g. perm1000 or perm10000)
    raw_perm_dirs = glob.glob(os.path.join(second_level_dir.format("glm"), f"vox{vox_thr_unc}_perm*"))
    i_fnames_unc = []
    if raw_perm_dirs:
        raw_perm_dir = raw_perm_dirs[0]
        for acq_name in config["design_exp"]["acq_names"]:
            tag_unc = "task-motor_acq-" + acq_name
            nii_cands_unc = glob.glob(os.path.join(raw_perm_dir, tag_unc, f"*_{tag_unc}_t.nii.gz"))
            if nii_cands_unc:
                i_fnames_unc.append(nii_cands_unc[0])
    if i_fnames_unc:
        z_slices_unc = [280, 266, 256, 243, 225]
        figures.plot_fmri_maps(
            i_fnames=i_fnames_unc,
            output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_uncorr_vox{vox_thr_unc}_avg_map.png"),
            stat_min=2, stat_max=5, cbar_label='t-value (uncorr.)', z_slices=z_slices_unc,
            background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
            underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]), redo=redo)
        figures.plot_fmri_maps_axial(
            i_fnames=i_fnames_unc,
            output_fname=os.path.join(output_fig, f"n{len(IDs)}_glm_uncorr_axial_vox{vox_thr_unc}_avg_map.png"),
            stat_min=2, stat_max=5, cbar_label='t-value (uncorr.)',
            background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
            underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]),
            z_slices=z_slices_unc, n_slices=len(z_slices_unc), redo=redo)
    else:
        print(f"INFO: No vox{vox_thr_unc} uncorrected GLM maps found — skipping.", flush=True)
except Exception as e:
    print(f"WARNING: GLM uncorrected group maps failed: {e}", flush=True)

# ------------------------------------------------------------------
# 6. ICC maps
# ------------------------------------------------------------------
print("", flush=True)
print("=== ICC maps ===", flush=True)
icc_base_dir = second_level_dir.format("icc")
z_slices_icc = [324, 306, 268, 238, 222, 204]
for icc_label, subdir in [
    ("run-01_vs_run-02",    "shimSlice_run01_vs_run02"),
    ("shimBase_vs_shimSlice", "shimBase_vs_shimSlice"),
]:
    icc_nii = os.path.join(icc_base_dir, subdir, "group_voxelwise_ICC.nii.gz")
    if not os.path.exists(icc_nii):
        print(f"INFO: No ICC map at {icc_nii}, skipping.", flush=True)
        continue
    try:
        icc_plot = figures.plot_fmri_maps(
            i_fnames=[icc_nii],
            output_fname=os.path.join(output_fig, f"n{len(IDs)}_icc_{icc_label}.png"),
            titles=[""], cmap="rainbow", cbar_label='ICC',
            z_slices=z_slices_icc, stat_min=0.3, stat_max=0.9,
            background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
            redo=redo)
        icc_axial_plot = figures.plot_fmri_maps_axial(
            i_fnames=[icc_nii],
            output_fname=os.path.join(output_fig, f"n{len(IDs)}_icc_{icc_label}_axial.png"),
            cmap="rainbow", stat_min=0.3, stat_max=0.9, titles=[""],
            background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
            underlay_fname=os.path.join(path_code, "template", config["PAM50_gm"]),
            z_slices=z_slices_icc, n_slices=len(z_slices_icc), redo=redo)
        figures.combine_plots(
            output_fname=os.path.join(output_fig, f"n{len(IDs)}_combined_icc_{icc_label}_plots.png"),
            map_files=[icc_plot], label_idx=False, axial_files=[icc_axial_plot],
            figsize=(2, 4), redo=redo)
    except Exception as e:
        print(f"WARNING: ICC {icc_label} figure failed: {e}", flush=True)


# ------------------------------------------------------------------
# 7. metrics extraction + comparison figures
# ------------------------------------------------------------------
print("", flush=True)
print("=== metrics extraction + comparison figures ===", flush=True)

from utils import compute_tsnr_map, compute_SNR

out_dir         = os.path.join(path_data, "derivatives", "processing", "figures", "metrics")
preprocessing_dir_compare = os.path.join(path_data, config["preprocess_dir"]["main_dir"])
os.makedirs(out_dir, exist_ok=True)

REGULAR_ACQS = config["design_exp"]["acq_names"]
SMOOTH_ACQS  = [k for k in config.get("derived_acq", {}) if "smooth3mm" in k]
AVG_ACQS     = [k for k in config.get("derived_acq", {}) if "avg3mm"    in k]
DERIVED_ACQS = SMOOTH_ACQS + AVG_ACQS
ALL_ACQS     = REGULAR_ACQS + DERIVED_ACQS

SHIM_TRIPLETS = [
    ("shimSlice+1mm+sms2", "shimSlice+1mm+sms2+smooth3mm", "shimSlice+3mm", "shimSlice"),
]
COLORS_3  = ["#E64B35", "#F39B7F", "#4DBBD5"]
XLABELS_3 = {
    "shimSlice+1mm+sms2":           "1mm\n(SMS2)",
    "shimSlice+1mm+sms2+smooth3mm": "1mm\n(smooth3mm)",
    "shimSlice+3mm":                "3mm",
}
Z_THRESHOLD = 3.1
MI_BINS = 32
pam50_mask_path = os.path.join(path_code, "template", config["PAM50_cord"])
all_tasks = config["design_exp"]["task_names"]


def _get_tsnr_for_derived(ID, task, acq_name):
    tag = f"task-{task}_acq-{acq_name}"
    moco_candidates = sorted(glob.glob(os.path.join(
        preprocessing_dir_compare.format(ID), "func", tag,
        "sct_fmri_moco", f"sub-{ID}_{tag}*_bold_moco.nii.gz"
    )))
    if not moco_candidates:
        return None
    moco_file = max(moco_candidates, key=lambda f: nib.load(f).shape[3])
    cache_dir = os.path.join(out_dir, "tsnr_maps", f"sub-{ID}", tag)
    return compute_tsnr_map(moco_file, cache_dir, redo)


def _get_seg(ID, task, acq_name):
    tag = f"task-{task}_acq-{acq_name}"
    return os.path.join(
        preprocessing_dir_compare.format(ID), "func", tag,
        f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz"
    )


def _get_moco_mean(ID, task, acq_name):
    tag = f"task-{task}_acq-{acq_name}"
    tag_dir = os.path.join(preprocessing_dir_compare.format(ID), "func", tag)
    cands = (glob.glob(os.path.join(tag_dir, "sct_fmri_moco", f"sub-{ID}_{tag}*_bold_moco_mean.nii.gz")) or
             glob.glob(os.path.join(tag_dir, f"sub-{ID}_{tag}*_bold_moco_mean.nii.gz")))
    return sorted(cands)[-1] if cands else None


def wilcoxon_str(a, b):
    stat, pval = stats.wilcoxon(a, b)
    if pval < 0.001:   label = "p < 0.001 ***"
    elif pval < 0.01:  label = f"p = {pval:.3f} **"
    elif pval < 0.05:  label = f"p = {pval:.3f} *"
    else:              label = f"p = {pval:.3f} ns"
    return stat, pval, label


def draw_bracket(ax, x1, x2, y_top, label, fontsize=8):
    tick = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
    ax.plot([x1, x1, x2, x2], [y_top - tick, y_top, y_top, y_top - tick], "k-", linewidth=1)
    ax.text((x1 + x2) / 2, y_top + tick * 0.5, label, ha="center", va="bottom", fontsize=fontsize)


def _triplet_fig(ax, d, acqs, xpos, common_all, common_23, pstr23, ylabel, title):
    vals_list = []
    for acq in acqs:
        subs = common_23 if acq != acqs[0] else common_all
        vals_list.append(d[acq].reindex(subs).dropna().values if len(subs) > 0 else np.array([]))
    bp_data = [v for v in vals_list if v.size > 0]
    bp_pos  = [xpos[i] for i, v in enumerate(vals_list) if v.size > 0]
    bp_cols = [COLORS_3[i] for i, v in enumerate(vals_list) if v.size > 0]
    bp = ax.boxplot(bp_data, positions=bp_pos, widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], bp_cols):
        patch.set_facecolor(col); patch.set_alpha(0.45)
    for i, (wh, ca) in enumerate(zip(bp["whiskers"], bp["caps"])):
        wh.set_color(bp_cols[i // 2]); wh.set_alpha(0.6)
        ca.set_color(bp_cols[i // 2]); ca.set_alpha(0.6)
    for sub in common_all:
        ax.plot(xpos, [d[a].get(sub, np.nan) for a in acqs],
                "o-", color="dimgray", alpha=0.5, linewidth=1, markersize=4, zorder=3)
    for sub in common_23.difference(common_all):
        ax.plot([1, 2], [d[acqs[1]][sub], d[acqs[2]][sub]], "o--",
                color="dimgray", alpha=0.4, linewidth=1, markersize=4, zorder=3)
    ax.set_ylim(bottom=0)
    y_ceil = max(max(v) for v in bp_data) * 1.25
    ax.set_ylim(top=y_ceil)
    draw_bracket(ax, 1, 2, y_ceil * 0.91, pstr23, fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels([XLABELS_3[a] for a in acqs], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-0.6, 2.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# --- 7a. tSNR within SC mask ---
tsnr_csv = os.path.join(out_dir, "tsnr_sc_mask.csv")
if not os.path.exists(tsnr_csv) or redo:
    records = []
    for ID in IDs:
        for task in ["rest"]:
            for acq_name in ALL_ACQS:
                tag = f"task-{task}_acq-{acq_name}"
                if acq_name in DERIVED_ACQS:
                    tsnr_map = _get_tsnr_for_derived(ID, task, acq_name)
                else:
                    cands = glob.glob(os.path.join(preprocessing_dir_compare.format(ID), "func", tag, "tsnr",
                                                   f"sub-{ID}_{tag}*_bold_moco_tsnr.nii.gz"))
                    tsnr_map = sorted(cands)[-1] if cands else None
                if not tsnr_map or not os.path.exists(tsnr_map):
                    continue
                seg_path = _get_seg(ID, task, acq_name)
                if not os.path.exists(seg_path):
                    continue
                vals = nib.load(tsnr_map).get_fdata()[nib.load(seg_path).get_fdata() > 0]
                if vals.size == 0:
                    continue
                ssnr_val = None
                moco_mean = _get_moco_mean(ID, task, acq_name)
                if moco_mean and os.path.exists(moco_mean):
                    try:
                        ssnr_val = float(compute_SNR(moco_mean, seg_path))
                    except Exception:
                        pass
                records.append({"subject": ID, "task": task, "acq": acq_name,
                                 "tsnr_sc": float(np.mean(vals)), "ssnr_sc": ssnr_val})
    pd.DataFrame(records).to_csv(tsnr_csv, index=False)
    print(f"Saved: {tsnr_csv}", flush=True)

try:
    df_tsnr = pd.read_csv(tsnr_csv)
except pd.errors.EmptyDataError:
    df_tsnr = pd.DataFrame()

# --- 7b. BOLD sensitivity ---
bold_csv = os.path.join(out_dir, "bold_sensitivity.csv")
glm_base_dir = os.path.join(path_data, config["first_level"]["dir"].format("glm", "").split("sub")[0])
if not os.path.exists(bold_csv) or redo:
    if not os.path.exists(pam50_mask_path):
        print(f"WARNING: PAM50 cord mask not found, skipping BOLD sensitivity.", flush=True)
        bold_df = pd.DataFrame()
    else:
        pam50_mask = nib.load(pam50_mask_path).get_fdata() > 0
        bold_records = []
        for ID in IDs:
            for task in all_tasks:
                for acq_name in ALL_ACQS:
                    tag = f"task-{task}_acq-{acq_name}"
                    zmap_cands = glob.glob(os.path.join(glm_base_dir, f"sub-{ID}", tag,
                                                        f"sub-{ID}_{tag}*trial_RH-rest*inTemplate.nii.gz"))
                    if not zmap_cands:
                        continue
                    zvals = nib.load(sorted(zmap_cands)[-1]).get_fdata()[pam50_mask]
                    zvals = zvals[np.isfinite(zvals)]
                    if zvals.size == 0:
                        continue
                    bold_records.append({"subject": ID, "task": task, "acq": acq_name,
                                         "peak_z": float(np.max(zvals)),
                                         "n_active": int(np.sum(zvals > Z_THRESHOLD))})
        bold_df = pd.DataFrame(bold_records)
        bold_df.to_csv(bold_csv, index=False)
        print(f"Saved: {bold_csv}", flush=True)
else:
    try:
        bold_df = pd.read_csv(bold_csv)
    except pd.errors.EmptyDataError:
        bold_df = pd.DataFrame()

# --- 7c. MI with T2* GRE ---
mi_csv   = os.path.join(out_dir, "mi_t2star.csv")
mi_cache = os.path.join(out_dir, "t2star_in_epi")
os.makedirs(mi_cache, exist_ok=True)

manual_dir_compare = os.path.join(path_data, config["manual_dir"])

def _compute_mi(x, y, bins=MI_BINS):
    hist2d, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = hist2d / hist2d.sum()
    px  = pxy.sum(axis=1, keepdims=True)
    py  = pxy.sum(axis=0, keepdims=True)
    mask = pxy > 0
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / (px * py)[mask])))

def _t2star_in_epi(ID, tag):
    """Register T2*w -> EPI space using centermass on SC segs, return path to warped T2*w."""
    tag_cache = os.path.join(mi_cache, f"sub-{ID}", tag)
    out_path  = os.path.join(tag_cache, f"sub-{ID}_T2star_in_EPI.nii.gz")
    if os.path.exists(out_path) and not redo:
        return out_path

    t2star = os.path.join(path_data, f"sub-{ID}", "anat", f"sub-{ID}_T2star.nii.gz")
    if not os.path.exists(t2star):
        return None

    # T2*w seg: prefer manual correction (derivatives/manual/ uses the BIDS label-SC
    # convention, see manual_label_filename() / #63-#64; preprocessing/ output naming
    # is unaffected)
    manual_t2star_seg = os.path.join(manual_dir_compare, f"sub-{ID}", "anat",
                                     manual_label_filename(f"sub-{ID}_T2star_seg.nii.gz", "SC"))
    auto_t2star_seg   = os.path.join(preprocessing_dir_compare.format(ID), "anat", "sct_deepseg",
                                     f"sub-{ID}_T2star_seg.nii.gz")
    t2star_seg = manual_t2star_seg if os.path.exists(manual_t2star_seg) else auto_t2star_seg
    if not os.path.exists(t2star_seg):
        return None

    tag_dir = os.path.join(preprocessing_dir_compare.format(ID), "func", tag)
    epi_seg_cands = (glob.glob(os.path.join(tag_dir, f"sub-{ID}_{tag}*_bold_moco_mean_seg.nii.gz")) or
                     glob.glob(os.path.join(tag_dir, "sct_deepseg", f"sub-{ID}_{tag}*_bold_moco_mean_seg.nii.gz")))
    epi_moco_cands = glob.glob(os.path.join(tag_dir, "sct_fmri_moco", f"sub-{ID}_{tag}*_bold_moco_mean.nii.gz"))
    if not epi_seg_cands or not epi_moco_cands:
        return None
    epi_seg  = sorted(epi_seg_cands)[0]
    epi_mean = sorted(epi_moco_cands)[-1]

    os.makedirs(tag_cache, exist_ok=True)
    cmd = (f"sct_register_multimodal -i {t2star} -iseg {t2star_seg}"
           f" -d {epi_mean} -dseg {epi_seg}"
           f" -param step=1,type=seg,algo=centermass"
           f" -ofolder {tag_cache} -x spline -v 0")
    ret = os.system(cmd)
    # sct_register_multimodal names output after the moving image basename
    t2star_base = os.path.basename(t2star).replace(".nii.gz", "")
    reg_out = os.path.join(tag_cache, f"{t2star_base}_reg.nii.gz")
    if ret != 0 or not os.path.exists(reg_out):
        return None
    os.rename(reg_out, out_path)
    return out_path

if not os.path.exists(mi_csv) or redo:
    mi_records = []
    for ID in IDs:
        for task in ["rest"]:
            for acq_name in ALL_ACQS:
                tag = f"task-{task}_acq-{acq_name}"
                tag_dir = os.path.join(preprocessing_dir_compare.format(ID), "func", tag)

                t2star_epi = _t2star_in_epi(ID, tag)
                if t2star_epi is None:
                    continue

                epi_moco_cands = glob.glob(os.path.join(tag_dir, "sct_fmri_moco",
                                                         f"sub-{ID}_{tag}*_bold_moco_mean.nii.gz"))
                epi_seg_cands  = (glob.glob(os.path.join(tag_dir, f"sub-{ID}_{tag}*_bold_moco_mean_seg.nii.gz")) or
                                  glob.glob(os.path.join(tag_dir, "sct_deepseg",
                                                         f"sub-{ID}_{tag}*_bold_moco_mean_seg.nii.gz")))
                if not epi_moco_cands or not epi_seg_cands:
                    continue

                epi_mean = sorted(epi_moco_cands)[-1]
                epi_seg  = sorted(epi_seg_cands)[0]
                epi_mask = nib.load(epi_seg).get_fdata() > 0
                if epi_mask.sum() < 10:
                    continue

                t2_vals  = nib.load(t2star_epi).get_fdata()[epi_mask]
                epi_vals = nib.load(epi_mean).get_fdata()[epi_mask]
                valid = np.isfinite(epi_vals) & np.isfinite(t2_vals) & (epi_vals > 0) & (t2_vals > 0)
                if valid.sum() < 10:
                    continue
                mi_records.append({"subject": ID, "task": task, "acq": acq_name,
                                   "mi": _compute_mi(epi_vals[valid], t2_vals[valid])})
    mi_df = pd.DataFrame(mi_records)
    mi_df.to_csv(mi_csv, index=False)
    print(f"Saved: {mi_csv}", flush=True)
else:
    try:
        mi_df = pd.read_csv(mi_csv)
    except pd.errors.EmptyDataError:
        mi_df = pd.DataFrame()

# --- 7d. tSNR triplet figures ---
for task in df_tsnr["task"].unique():
    df_task = df_tsnr[df_tsnr["task"] == task]
    for acq1, acq2, acq3, shim_label in SHIM_TRIPLETS:
        fig_path = os.path.join(output_fig_preprocessing, f"tsnr_sc_{shim_label}_{task}_triplet.png")
        if os.path.exists(fig_path) and not redo:
            print(f"Figure already exists: {fig_path}", flush=True)
            continue
        d = {a: df_task[df_task["acq"] == a].set_index("subject")["tsnr_sc"] for a in [acq1, acq2, acq3]}
        common_all = d[acq1].index.intersection(d[acq2].index).intersection(d[acq3].index)
        common_23  = d[acq2].index.intersection(d[acq3].index)
        if len(common_23) < 2:
            print(f"WARNING: Only {len(common_23)} paired for tSNR {shim_label} task-{task}, skipping.", flush=True)
            continue
        _, _, pstr23 = wilcoxon_str(d[acq2].loc[common_23].values, d[acq3].loc[common_23].values)
        try:
            fig, ax = plt.subplots(figsize=(4, 4.5))
            _triplet_fig(ax, d, [acq1, acq2, acq3], [0, 1, 2], common_all, common_23, pstr23,
                         "Mean tSNR within SC mask",
                         f"tSNR: 1mm vs 3mm  ({shim_label}, task-{task})\nn={len(common_23)} paired")
            fig.tight_layout(); fig.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
            print(f"Saved: {fig_path}", flush=True)
        except Exception as e:
            print(f"WARNING: tSNR triplet ({shim_label} task-{task}): {e}", flush=True)

# --- 7e. BOLD sensitivity triplet figures ---
if not bold_df.empty:
    for metric, ylabel in [("peak_z", "Peak z-score within PAM50 SC mask"),
                            ("n_active", f"Suprathreshold voxels (z > {Z_THRESHOLD})")]:
        for task in bold_df["task"].unique():
            bdf_task = bold_df[bold_df["task"] == task]
            for acq1, acq2, acq3, shim_label in SHIM_TRIPLETS:
                fig_path = os.path.join(output_fig_preprocessing, f"{metric}_{shim_label}_{task}_triplet.png")
                if os.path.exists(fig_path) and not redo:
                    print(f"Figure already exists: {fig_path}", flush=True)
                    continue
                d = {a: bdf_task[bdf_task["acq"] == a].set_index("subject")[metric] for a in [acq1, acq2, acq3]}
                common_all = d[acq1].index.intersection(d[acq2].index).intersection(d[acq3].index)
                common_23  = d[acq2].index.intersection(d[acq3].index)
                if len(common_23) < 2:
                    print(f"WARNING: Only {len(common_23)} paired for {metric} {shim_label} task-{task}, skipping.", flush=True)
                    continue
                _, _, pstr23 = wilcoxon_str(d[acq2].loc[common_23].values, d[acq3].loc[common_23].values)
                try:
                    fig, ax = plt.subplots(figsize=(4, 4.5))
                    _triplet_fig(ax, d, [acq1, acq2, acq3], [0, 1, 2], common_all, common_23, pstr23, ylabel,
                                 f"{metric}: 1mm vs 3mm  ({shim_label}, task-{task})\nn={len(common_23)} paired")
                    fig.tight_layout(); fig.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
                    print(f"Saved: {fig_path}", flush=True)
                except Exception as e:
                    print(f"WARNING: {metric} triplet ({shim_label} task-{task}): {e}", flush=True)

# --- 7f. MI 4-condition figure ---
MI_4COND = [
    ("shimBase+3mm",               "rest", 0,    "#2166AC", "shimBase\n3mm"),
    ("shimSlice+3mm",              "rest", 1,    "#74ADD1", "shimSlice\n3mm"),
    ("shimBase+1mm+sms2+avg3mm",   "rest", 2.5,  "#D73027", "shimBase\n1mm\n(avg3mm)"),
    ("shimSlice+1mm+sms2+avg3mm",  "rest", 3.5,  "#F4A582", "shimSlice\n1mm\n(avg3mm)"),
]
fig_path_4 = os.path.join(output_fig_preprocessing, "mi_t2star_4cond.png")
if not mi_df.empty and (not os.path.exists(fig_path_4) or redo):
    try:
        mi_ser = {c[0]: mi_df[(mi_df["task"] == c[1]) & (mi_df["acq"] == c[0])].set_index("subject")["mi"]
                  for c in MI_4COND}
        base3_acq, slice3_acq = MI_4COND[0][0], MI_4COND[1][0]
        base1_acq, slice1_acq = MI_4COND[2][0], MI_4COND[3][0]
        common_3mm       = mi_ser[base3_acq].index.intersection(mi_ser[slice3_acq].index)
        common_1mm       = mi_ser[base1_acq].index.intersection(mi_ser[slice1_acq].index)
        common_slice_res = mi_ser[slice3_acq].index.intersection(mi_ser[slice1_acq].index)

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        xpos_list = [c[2] for c in MI_4COND]; color_list = [c[3] for c in MI_4COND]
        label_list = [c[4] for c in MI_4COND]
        vals_all   = [mi_ser[c[0]].dropna().values for c in MI_4COND]
        bp = ax.boxplot(vals_all, positions=xpos_list, widths=0.45, patch_artist=True,
                        showfliers=False, medianprops=dict(color="white", linewidth=2))
        for patch, col in zip(bp["boxes"], color_list):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        for i, (wh, ca) in enumerate(zip(bp["whiskers"], bp["caps"])):
            wh.set_color(color_list[i // 2]); wh.set_alpha(0.6)
            ca.set_color(color_list[i // 2]); ca.set_alpha(0.6)
        for sub in common_3mm:
            ax.plot([0, 1], [mi_ser[base3_acq][sub], mi_ser[slice3_acq][sub]],
                    "o-", color="dimgray", alpha=0.5, linewidth=1, markersize=4, zorder=3)
        for sub in common_1mm:
            ax.plot([2.5, 3.5], [mi_ser[base1_acq][sub], mi_ser[slice1_acq][sub]],
                    "o-", color="dimgray", alpha=0.5, linewidth=1, markersize=4, zorder=3)
        ax.set_ylim(bottom=0)
        y_max = max(v.max() for v in vals_all if v.size > 0) * 1.50
        ax.set_ylim(top=y_max)
        bracket_inner_y, bracket_outer_y = y_max * 0.83, y_max * 0.96
        for (a, b, common), (bx1, bx2), by in [
            ((base3_acq,  slice3_acq, common_3mm),       (0,   1),   bracket_inner_y),
            ((base1_acq,  slice1_acq, common_1mm),       (2.5, 3.5), bracket_inner_y),
            ((slice3_acq, slice1_acq, common_slice_res), (1,   3.5), bracket_outer_y),
        ]:
            if len(common) >= 2:
                _, _, ps = wilcoxon_str(mi_ser[a].loc[common].values, mi_ser[b].loc[common].values)
                draw_bracket(ax, bx1, bx2, by, ps, fontsize=8)
        ax.set_xticks(xpos_list); ax.set_xticklabels(label_list, fontsize=8.5)
        ax.set_ylabel("Mutual Information with T2*w GRE\n(native EPI space, within SC mask)", fontsize=9)
        ax.set_title("MI with T2*w: shimBase vs shimSlice\n(3mm and 1mm acquisitions)", fontsize=9)
        ax.set_xlim(-0.6, 4.1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout(); fig.savefig(fig_path_4, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {fig_path_4}", flush=True)
    except Exception as e:
        print(f"WARNING: MI 4-condition figure failed: {e}", flush=True)

# --- 7g. tSNR / sSNR 4-condition figures (matching MI style) ---
SNRPLOT_4COND = [
    ("shimBase+3mm",                  "rest", 0,    "#2166AC", "shimBase\n3mm"),
    ("shimSlice+3mm",                 "rest", 1,    "#74ADD1", "shimSlice\n3mm"),
    ("shimBase+1mm+sms2+smooth3mm",   "rest", 2.5,  "#D73027", "shimBase\n1mm\n(smooth3mm)"),
    ("shimSlice+1mm+sms2+smooth3mm",  "rest", 3.5,  "#F4A582", "shimSlice\n1mm\n(smooth3mm)"),
]
for snr_metric, col_name, ylabel in [
    ("tSNR", "tsnr_sc", "Mean tSNR within SC mask"),
    ("sSNR", "ssnr_sc", "Mean sSNR within SC mask"),
]:
    fig_path = os.path.join(output_fig_preprocessing, f"{len(IDs)}_{snr_metric.lower()}_4cond.png")
    if df_tsnr.empty or col_name not in df_tsnr.columns:
        print(f"INFO: No {col_name} data, skipping {snr_metric} 4-condition plot.", flush=True)
        continue
    if os.path.exists(fig_path) and not redo:
        print(f"Figure already exists: {fig_path}", flush=True)
        continue
    try:
        snr_ser = {c[0]: df_tsnr[(df_tsnr["task"] == c[1]) & (df_tsnr["acq"] == c[0])
                                 ].set_index("subject")[col_name].dropna()
                   for c in SNRPLOT_4COND}
        base3_acq, slice3_acq = SNRPLOT_4COND[0][0], SNRPLOT_4COND[1][0]
        base1_acq, slice1_acq = SNRPLOT_4COND[2][0], SNRPLOT_4COND[3][0]
        common_3mm       = snr_ser[base3_acq].index.intersection(snr_ser[slice3_acq].index)
        common_1mm       = snr_ser[base1_acq].index.intersection(snr_ser[slice1_acq].index)
        common_slice_res = snr_ser[slice3_acq].index.intersection(snr_ser[slice1_acq].index)

        xpos_list  = [c[2] for c in SNRPLOT_4COND]
        color_list = [c[3] for c in SNRPLOT_4COND]
        label_list = [c[4] for c in SNRPLOT_4COND]
        vals_all   = [snr_ser[c[0]].values for c in SNRPLOT_4COND]

        if all(v.size == 0 for v in vals_all):
            print(f"INFO: No {snr_metric} values found, skipping.", flush=True)
            continue

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        bp = ax.boxplot(vals_all, positions=xpos_list, widths=0.45, patch_artist=True,
                        showfliers=False, medianprops=dict(color="white", linewidth=2))
        for patch, col in zip(bp["boxes"], color_list):
            patch.set_facecolor(col); patch.set_alpha(0.5)
        for i, (wh, ca) in enumerate(zip(bp["whiskers"], bp["caps"])):
            wh.set_color(color_list[i // 2]); wh.set_alpha(0.6)
            ca.set_color(color_list[i // 2]); ca.set_alpha(0.6)
        for sub in common_3mm:
            ax.plot([0, 1], [snr_ser[base3_acq][sub], snr_ser[slice3_acq][sub]],
                    "o-", color="dimgray", alpha=0.5, linewidth=1, markersize=4, zorder=3)
        for sub in common_1mm:
            ax.plot([2.5, 3.5], [snr_ser[base1_acq][sub], snr_ser[slice1_acq][sub]],
                    "o-", color="dimgray", alpha=0.5, linewidth=1, markersize=4, zorder=3)
        ax.set_ylim(bottom=0)
        y_max = max(v.max() for v in vals_all if v.size > 0) * 1.50
        ax.set_ylim(top=y_max)
        bracket_inner_y, bracket_outer_y = y_max * 0.83, y_max * 0.96
        for (a, b, common), (bx1, bx2), by in [
            ((base3_acq,  slice3_acq, common_3mm),       (0,   1),   bracket_inner_y),
            ((base1_acq,  slice1_acq, common_1mm),       (2.5, 3.5), bracket_inner_y),
            ((slice3_acq, slice1_acq, common_slice_res), (1,   3.5), bracket_outer_y),
        ]:
            if len(common) >= 2:
                _, _, ps = wilcoxon_str(snr_ser[a].loc[common].values, snr_ser[b].loc[common].values)
                draw_bracket(ax, bx1, bx2, by, ps, fontsize=8)
        ax.set_xticks(xpos_list); ax.set_xticklabels(label_list, fontsize=8.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{snr_metric}: shimBase vs shimSlice\n(3mm and 1mm acquisitions, REST)", fontsize=9)
        ax.set_xlim(-0.6, 4.1)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        fig.tight_layout(); fig.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)
        print(f"Saved: {fig_path}", flush=True)
    except Exception as e:
        print(f"WARNING: {snr_metric} 4-condition figure failed: {e}", flush=True)

print("", flush=True)
print("=== figures_workflow done ===", flush=True)
