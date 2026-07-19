    #!/usr/bin/env python
# coding: utf-8

# # Spinal cord fMRI second level
#
# Description: This workflow provides code for second level analyses 
# I. Run second level glm analysis for each subject and task
# II. Run ICC analyses
#------------------------------------------------------------------
#------ Initialization
#------------------------------------------------------------------
# Main imports ------------------------------------------------------------
import re, json, sys, os, glob, argparse
import pandas as pd

# Get the environment variable PATH_CODE
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(path_code, "config", "config_spine_7t_fmri.json")) as config_file: # the notebook should be in 'xx/notebook/' folder #config_proprio
    config = json.load(config_file) # load config file should be open first and the path inside modified

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--tasks", nargs='+', default=[""])
parser.add_argument("--verbose", default="False")
parser.add_argument("--redo", default="True")
parser.add_argument("--path-data", required=True)
parser.add_argument("--n-perm", type=int, default=1000, help="Number of permutations for non-parametric GLM (default: 1000)")
parser.add_argument("--n-jobs", type=int, default=10, help="Number of parallel jobs for permutation testing (default: 10)")
args = parser.parse_args()

IDs = args.ids
tasks = args.tasks
verbose = args.verbose.lower() == "true"
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)
n_perm = args.n_perm
n_jobs = args.n_jobs

config["raw_dir"]=path_data
config["code_dir"]=path_code

participants_tsv = pd.read_csv(os.path.join(path_code, "config", "participants.tsv"), sep='\t', dtype={'participant_id': str})

new_IDs=[]
if IDs == [""]:
    for ID in participants_tsv["participant_id"]:
        new_IDs.append(ID)

    IDs = new_IDs

if tasks != [""]:
    config["design_exp"]["task_names"] = tasks

#Import scripts
sys.path.append(os.path.join(path_code, "code")) # Change this line according to your directory
import postprocess, preprocess

glm_ana = postprocess.GLM_main(config,IDs=IDs)
preprocess_Sc = preprocess.Preprocess_Sc(config,IDs=IDs)
tsnr_ana = postprocess.TSNR_main(config, IDs,redo)

# initialize directories
preprocessing_dir = os.path.join(config["raw_dir"], config["preprocess_dir"]["main_dir"])
first_level_dir = os.path.join(config["raw_dir"], config["first_level"]["dir"])
second_level_dir = os.path.join(config["raw_dir"], config["second_level"]["dir"])

mask = os.path.join(first_level_dir.format('glm',"").split("sub")[0], "common_mask_PAM50.nii.gz")


def find_tsnr_dir_for_acq(ID, acq_name):
    """
    Find the func/<tag>/tsnr/ folder for this subject/acquisition (preferring a
    "rest" tag match, falling back to any tag match, same selection logic as
    before). tSNR now lives alongside each acquisition's other preprocessing
    outputs (see TSNR_main.tsnr_dir_for() in postprocess.py), not under
    first_level/ (#70).

    Returns (tag, tsnr_dir), or (None, None) if no match is found.
    """
    func_dir = os.path.join(preprocessing_dir.format(ID), "func")
    if not os.path.isdir(func_dir):
        return None, None
    dirs = [d for d in os.listdir(func_dir)
            if os.path.isdir(os.path.join(func_dir, d, "tsnr"))]
    rest_dirs = [d for d in dirs if "rest" in d and acq_name in d]
    selected_dirs = rest_dirs if rest_dirs else [d for d in dirs if acq_name in d]
    if not selected_dirs:
        return None, None
    return selected_dirs[0], os.path.join(func_dir, selected_dirs[0], "tsnr")


#------------------------------------------------------------------
#------ Compute average tSNR
#------------------------------------------------------------------
for acq_name in config["design_exp"]["acq_names"]:
    tsnr_id_fname = []
    cord_seg_file = []
    warp_file = []
    valid_IDs = []
    for ID in IDs:
        tag, tsnr_dir = find_tsnr_dir_for_acq(ID, acq_name)
        if tsnr_dir is None:
            print(f"WARNING: No tSNR dir for sub-{ID} acq-{acq_name}, skipping.", flush=True)
            continue
        tsnr_files = glob.glob(os.path.join(tsnr_dir, "*_moco_tsnr.nii.gz"))
        seg_files = glob.glob(os.path.join(preprocessing_dir.format(ID), 'func', tag, config["preprocess_f"]["func_seg"].format(ID, tag, "")))
        warp_files = glob.glob(os.path.join(preprocessing_dir.format(ID), 'func', tag, f"sub-{ID}_{tag}_from-func_to_PAM50_mode-image_xfm.nii.gz"))
        if not (tsnr_files and seg_files and warp_files):
            print(f"WARNING: Missing tSNR/seg/warp for sub-{ID} acq-{acq_name}, skipping.", flush=True)
            continue
        task_name = tag.split("_")[0].split("-")[1]
        tsnr_id_fname.append(tsnr_files[0])
        cord_seg_file.append(seg_files[0])
        warp_file.append(warp_files[0])
        valid_IDs.append(ID)

    if not valid_IDs:
        print(f"WARNING: No subjects with tSNR data for acq-{acq_name}, skipping.", flush=True)
        continue

    fname_avg_tsnr = tsnr_ana.generate_average_tsnr_in_pam50(
        IDs=valid_IDs,
        task_name=task_name,
        acq_name=acq_name,
        tsnr_fnames=tsnr_id_fname,
        seg_fnames=cord_seg_file,
        warp_fnames=warp_file,
        fname_mask=mask)

    # Boxplot figures moved to figures_workflow.py (run with --figures).

# Also average tSNR for derived (avg3mm) acquisitions, needed for the 3mm vs avg3mm map comparison
avg_acq_names = {k: v for k, v in config.get("derived_acq", {}).items() if "n_slices_avg" in v}
tsnr_avg3mm = {}  # acq_name -> fname_avg_tsnr in PAM50
for acq_name in avg_acq_names:
    source_acq = avg_acq_names[acq_name]["source_acq"]
    tsnr_id_fname = []
    cord_seg_file = []
    warp_file = []
    valid_IDs = []
    for ID in IDs:
        tag, tsnr_dir = find_tsnr_dir_for_acq(ID, acq_name)
        if tsnr_dir is None:
            continue
        tsnr_files = glob.glob(os.path.join(tsnr_dir, "*_moco_tsnr.nii.gz"))
        # seg and warp live under the source acq's preprocessing dir (derived has none)
        source_tag = f"task-rest_acq-{source_acq}"
        seg_files = glob.glob(os.path.join(preprocessing_dir.format(ID), 'func', source_tag, config["preprocess_f"]["func_seg"].format(ID, source_tag, "")))
        warp_files = glob.glob(os.path.join(preprocessing_dir.format(ID), 'func', source_tag, f"sub-{ID}_{source_tag}_from-func_to_PAM50_mode-image_xfm.nii.gz"))
        if not (tsnr_files and seg_files and warp_files):
            continue
        task_name = tag.split("_")[0].split("-")[1]
        tsnr_id_fname.append(tsnr_files[0])
        cord_seg_file.append(seg_files[0])
        warp_file.append(warp_files[0])
        valid_IDs.append(ID)
    if not valid_IDs:
        print(f"INFO: No subjects with avg3mm tSNR data for acq-{acq_name}, skipping.", flush=True)
        continue
    tsnr_avg3mm[acq_name] = tsnr_ana.generate_average_tsnr_in_pam50(
        IDs=valid_IDs, task_name=task_name, acq_name=acq_name,
        tsnr_fnames=tsnr_id_fname, seg_fnames=cord_seg_file,
        warp_fnames=warp_file, fname_mask=mask)

#------------------------------------------------------------------
#------ Compute average FD
#------------------------------------------------------------------
os.makedirs(second_level_dir.format("FD"), exist_ok=True)
fd_files=[]
for acq_name in config["design_exp"]["acq_names"]:
    run_names=[]
    valid_IDs_fd=[]
    tag="task-motor" + "_acq-" + acq_name

    for ID in IDs:
        raw_func=sorted(glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz')))
        if not raw_func:
            continue
        match = re.search(r"_?(run-\d+)", raw_func[0])
        run_names.append(match.group(1) if match else "")
        valid_IDs_fd.append(ID)

    if not valid_IDs_fd:
        print(f"WARNING: No subjects with data for FD {tag}, skipping.", flush=True)
        continue
    output_file=second_level_dir.format("FD") + f"/n{len(valid_IDs_fd)}_{tag}_FD.csv"
    fd_files.append(glm_ana.extract_FD(IDs=valid_IDs_fd,task_name=tag,run_name=run_names,output_file=output_file,redo=False))

postprocess.pair_ttest(
    csv_files=fd_files,
    value_col='mean_FD',
    conds=[('shimSlice+3mm', 'shimSlice+1mm+sms2')],
    acq_col='acq',
    task_filter=None, task_col=None,
    output_fname=output_file.split('.csv')[0]+"_stats.csv", redo=True)

#------------------------------------------------------------------
#------ Run second level analysis
#------------------------------------------------------------------

print("")
print("=== Second level analysis script Start ===", flush=True)
print("Number of Participant included : ", len(IDs), flush=True)
print("===================================", flush=True)
print("")

common_mask_fname = os.path.join(first_level_dir.split("sub")[0], "common_mask_PAM50.nii.gz").format("glm")

import time as _time

values_csv_pair={};metrics_csv_pair={}
for cluster_corr in [0.01,0.001]:
    values_csv_pair[cluster_corr]={};metrics_csv_pair[cluster_corr]={}
    for vox_thr in [0.005]:
        values_csv_pair[cluster_corr][vox_thr]=[];metrics_csv_pair[cluster_corr][vox_thr]=[]
        for task_name in ["motor"]:
            for acq_name in config["design_exp"]["acq_names"]:
                i_fnames=[]
                tag = "task-" + task_name + "_acq-" + acq_name
                for ID in IDs:
                    raw_func = sorted(glob.glob(os.path.join(config["raw_dir"], f'sub-{ID}', 'func', f'sub-{ID}_{tag}_*bold.nii.gz')))
                    if not raw_func:
                        continue
                    match = re.search(r"_?(run-\d+)", raw_func[0])
                    run_name = match.group(1) if match else ""
                    glm_matches = glob.glob(os.path.join(first_level_dir.format('glm',ID), f"{tag}", f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"))
                    if glm_matches:
                        i_fnames.append(glm_matches[0])

                if len(i_fnames) < 2:
                    print(f"WARNING: Only {len(i_fnames)} subject(s) for {tag} — need at least 2 for second-level GLM, skipping.", flush=True)
                    continue

                _t_task = _time.time()
                print(f"--- Starting second-level GLM: {tag} | cluster_corr={cluster_corr} vox_thr={vox_thr} | N={len(i_fnames)} subjects ---", flush=True)

                z_map_file = glm_ana.run_second_level_glm(i_fnames=i_fnames,
                                                                mask_fname=common_mask_fname,
                                                                task_name=tag,
                                                                run_name="",
                                                                parametric=False,
                                                                n_perm=n_perm,
                                                                n_jobs=n_jobs,
                                                                vox_thr=vox_thr,
                                                                cluster_corr=cluster_corr,
                                                                redo=redo,
                                                                verbose=verbose)

                metrics_csv,values_csv = glm_ana.extract_metrics(i_fname=z_map_file,threshold=0)
                metrics_csv_pair[cluster_corr][vox_thr].append(metrics_csv)
                values_csv_pair[cluster_corr][vox_thr].append(values_csv)

                print(metrics_csv_pair)

                print("")
                print(f'=== Second level done for : {tag}, cluster: {cluster_corr} vox: {vox_thr} | elapsed: {_time.time()-_t_task:.1f}s ===', flush=True)
                print("=========================================", flush=True)

# Figure generation moved to figures_workflow.py (run with --figures).

#------------------------------------------------------------------
#------ compute test-retest reproductibility using ICC
#------------------------------------------------------------------

# ----------  between shimSlice run01 vs run02 ---
print("", flush=True)
print(f'=== ICC between sliceShim run-01 and run-02  start', flush=True)
print("=========================================", flush=True)
output_dir = os.path.join(second_level_dir.format("icc"), "shimSlice_run01_vs_run02")
os.makedirs(output_dir, exist_ok=True)
i_fnames_by_runs = []
tag = "task-motor_acq-shimSlice+3mm"
IDs_2runs = []
for ID in IDs:
    raw_func = sorted(glob.glob(os.path.join( config["raw_dir"], f"sub-{ID}", "func", f"sub-{ID}_{tag}_*bold.nii.gz")))
    
    # Only keep participants with 2 runs
    if len(raw_func) != 2:
        continue
    IDs_2runs.append(ID)
    i_fnames_runs = []
    for fname in raw_func:
        run_name = re.search(r"_?(run-\d+)", fname).group(1)
        stat_map = glob.glob(os.path.join(
            first_level_dir.format("glm",ID), tag, f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"
        ))[0]
        i_fnames_runs.append(stat_map)
    
    i_fnames_by_runs.append(i_fnames_runs)

if not IDs_2runs:
    print("WARNING: No subjects with 2 runs of shimSlice+3mm — ICC run-01 vs run-02 skipped.", flush=True)
else:
    try:
        glm_ana.run_icc(IDs=IDs_2runs, i_fnames=i_fnames_by_runs, o_dir=output_dir, mask_file=mask, threshold=0, redo=redo)
    except Exception as e:
        print(f"WARNING: ICC run-01 vs run-02 failed: {e}", flush=True)

print("", flush=True)
print(f'=== ICC between sliceShim run-01 and run-02  done', flush=True)
print("=========================================", flush=True)

# ----------  between shimBase and shimSlice ---
print("", flush=True)
print(f'=== ICC between sliceShim and sliceBase  start', flush=True)
print("=========================================", flush=True)
if not IDs_2runs:
    print("WARNING: No subjects with 2 runs — ICC shimBase vs shimSlice skipped.", flush=True)
else:
    output_dir = os.path.join(second_level_dir.format("icc"), "shimBase_vs_shimSlice")
    os.makedirs(output_dir, exist_ok=True)
    i_fnames_by_runs = []
    for ID in IDs_2runs:
        i_fnames_runs = []
        for acq_name in config["design_exp"]["acq_names"]:
            tag = "task-motor" + "_acq-" + acq_name
            raw_func = sorted(glob.glob(os.path.join(config["raw_dir"], f"sub-{ID}", "func", f"sub-{ID}_{tag}_*bold.nii.gz")))
            if not raw_func:
                continue
            match = re.search(r"_?(run-\d+)", raw_func[0])
            run_name = match.group(1) if match else ""
            glm_matches = glob.glob(os.path.join(first_level_dir.format("glm", ID), tag, f"*{tag}*{run_name}*trial_RH-rest*inTemplate.nii.gz"))
            if glm_matches:
                i_fnames_runs.append(glm_matches[0])
        if i_fnames_runs:
            i_fnames_by_runs.append(i_fnames_runs)

    try:
        icc_maps, icc_maps_s = glm_ana.run_icc(IDs=IDs_2runs, i_fnames=i_fnames_by_runs, o_dir=output_dir, mask_file=mask, threshold=0)
    except Exception as e:
        print(f"WARNING: ICC shimBase vs shimSlice failed: {e}", flush=True)

print("", flush=True)
print(f'=== ICC between sliceShim and sliceBase  done', flush=True)
print("=========================================", flush=True)
