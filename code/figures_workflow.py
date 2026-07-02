#!/usr/bin/env python
# coding: utf-8

# Generate all publication figures from pre-computed results.
# Run after firstlevel_workflow, secondlevel_workflow, and compare_workflow have completed.

import re, json, sys, os, glob, argparse
import pandas as pd

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
import figures as figures_module

figures = figures_module.Figures_main(config, IDs=IDs)

# Directories
first_level_dir  = os.path.join(config["raw_dir"], config["first_level"]["dir"])
second_level_dir = os.path.join(config["raw_dir"], config["second_level"]["dir"])
output_fig = os.path.join(config["raw_dir"], config["figures_dir"]["main_dir"], "second_level")
os.makedirs(output_fig, exist_ok=True)

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

i_fnames_by_runs = []
for ID in IDs:
    i_fnames_runs = []
    for task_name in config["design_exp"]["task_names"]:
        for acq_name in config["design_exp"]["acq_names"]:
            tag = "task-" + task_name + "_acq-" + acq_name
            raw_func = sorted(glob.glob(os.path.join(
                config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz'
            )))
            if not raw_func:
                continue
            if len(raw_func) == 2 and tag == "task-motor_acq-shimSlice+3mm":
                for fname in raw_func:
                    match = re.search(r"_?(run-\d+)", fname)
                    run_name = match.group(1)
                    matches = glob.glob(os.path.join(
                        first_level_dir.format("glm", ID), tag,
                        f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"
                    ))
                    if matches:
                        i_fnames_runs.append(matches[0])
            else:
                match = re.search(r"_?(run-\d+)", raw_func[0])
                run_name = match.group(1) if match else ""
                matches = glob.glob(os.path.join(
                    first_level_dir.format("glm", ID), tag,
                    f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"
                ))
                if matches:
                    i_fnames_runs.append(matches[0])
    if i_fnames_runs:
        i_fnames_by_runs.append(i_fnames_runs)

try:
    figures.plot_first_level_maps(
        i_fnames=i_fnames_by_runs,
        output_fname=os.path.join(fig_dir_first, f"first_level_task_by_runs_n{len(i_fnames_by_runs)}.png"),
        background_fname=os.path.join(path_code, "template", config["PAM50_t2"]),
        mask_fname=common_mask_fname,
        titles=["shimBase", "shimSlice", "shimSlice"],
        task_name=tag,
        participant_ids=IDs,
        verbose=True,
        redo=redo)
except Exception as e:
    print(f"WARNING: First-level maps skipped: {e}", flush=True)

# ------------------------------------------------------------------
# 3. tSNR boxplots
# ------------------------------------------------------------------
print("", flush=True)
print("=== tSNR / sSNR boxplots ===", flush=True)
tsnr_dir = os.path.join(
    config["raw_dir"],
    config["first_level"]["dir"].format("snr", "").split("sub")[0]
)
for metric in ["tsnr", "ssnr"]:
    csv_matches  = glob.glob(os.path.join(tsnr_dir, f"{metric}*_metrics_reduced.csv"))
    stat_matches = glob.glob(os.path.join(tsnr_dir, f"{metric}*_metrics_reduced_stats.csv"))
    if not csv_matches or not stat_matches:
        print(f"INFO: No {metric} CSV found, skipping boxplot.", flush=True)
        continue
    (ymin, ymax) = (6, 17) if metric == "tsnr" else (1.5, 4.5)
    y_label = "temporal SNR" if metric == "tsnr" else "spatial SNR"
    try:
        figures.boxplots(
            csv_file=csv_matches[0],
            output_fname=os.path.join(output_fig, f"{len(IDs)}_{metric}_boxplot.png"),
            ymin=ymin, ymax=ymax, stats_file=stat_matches[0],
            specify_y_label=y_label,
            x_data="acq", x_order=["shimBase", "shimSlice"],
            indiv_values=True,
            y_data=metric, redo=redo)
    except Exception as e:
        print(f"WARNING: {metric} boxplot failed: {e}", flush=True)

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

print("", flush=True)
print("=== figures_workflow done ===", flush=True)
