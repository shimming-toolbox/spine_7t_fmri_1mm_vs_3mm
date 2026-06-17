#!/usr/bin/env python
# coding: utf-8

#  Spinal Cord fMRI preprocessing
# ____________________________________________________
#
# ### Project: acdc_spine_7T
# ____________________________________________________
# @ author: Caroline Landelle, caroline.landelle@mcgill.ca // landelle.caroline@gmail.com
# July 2025
#
# Description:
# This notebook provides code for preprocessing fMRI data of spinal cord acquisition at 7T.
#
# Toolbox required:
# > SpinalCordToolbox
# > FSL (Python)
#
# ____________________________________________________
#
# nb: The Philips system includes additional "dummy scans" at the beginning of the acquisition to allow the magnetization to stabilize to a steady state. The dummy scans are not stored, so they will make the banging sound like normal scans but there will be no data associated with them.
#
#------------------------------------------------------------------
#------ Initialization
#------------------------------------------------------------------
# Imports
import sys, json, glob, os, re, shutil, argparse
import pandas as pd

# get path of the parent location of this file, and go up one level
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(path_code, "code"))  # Change this line according to your directory
from preprocess import Preprocess_main, Preprocess_Sc, copy_warping_fields_from_ref_tag, copy_segmentation_from_ref_tag
from preprocess import copy_csf_segmentation_from_ref_tag
import utils

with open(os.path.join(path_code, "config", "config_spine_7t_fmri.json")) as config_file:
    config = json.load(config_file)  # load config file should be open first and the path inside modified

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--tasks", nargs='+', default=[""])
parser.add_argument("--verbose", default="False")
parser.add_argument("--manual_centerline", default="False")
parser.add_argument("--auto_vert_labels", default="True")
parser.add_argument("--redo", default="True")
parser.add_argument("--path-data", required=True)
args = parser.parse_args()

IDs = args.ids
tasks = args.tasks
verbose = args.verbose.lower() == "true"
manual_centerline = args.manual_centerline.lower() == "true"
auto_vert_labels = args.auto_vert_labels.lower() == "true"
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)

config["raw_dir"] = path_data
config["code_dir"] = path_code

# Display parameters in print
print("=== Preprocessing parameters ===", flush=True)
print("Participant IDs: ", IDs, flush=True)
print("Tasks to process: ", tasks, flush=True)
print("Verbose: ", verbose, flush=True)
print("Manual centerline: ", manual_centerline, flush=True)
print("Auto vertebral labels: ", auto_vert_labels, flush=True)
print("Redo steps: ", redo, flush=True)
print("================================", flush=True)

# Load participants info
participants_tsv = pd.read_csv(os.path.join(path_code, 'config', 'participants.tsv'), sep='\t',dtype={'participant_id': str})
acq_parameters = []

new_IDs=[]
if IDs == [""]:
    for ID in participants_tsv["participant_id"]:
        new_IDs.append(ID)
    IDs = new_IDs

utils.print_participant_metrics(participants_tsv, IDs)

if tasks != [""]:
    config["design_exp"]["task_names"] = tasks

#Initialize codes
Preprocess_main = Preprocess_main(config, IDs=IDs) # initialize the function
preprocess_Sc = Preprocess_Sc(config, IDs=IDs) # initialize the function
ses_name = ""

# initialize directories
preprocessing_dir = os.path.join(config["raw_dir"], config["preprocess_dir"]["main_dir"])
derivatives_dir = os.path.join(config["raw_dir"], config["derivatives_dir"])
manual_dir = os.path.join(config["raw_dir"], config["manual_dir"])



def epi_full_processing(ID, func_file, tag, manual_centerline, warpT2w_PAM50_files, params_moco, o_dir, redo, verbose):
    # ------------------------------------------------------------------
    # ------ Create mask around the cord for moco
    # ------------------------------------------------------------------
    o_img = os.path.join(o_dir, os.path.basename(func_file).split(".")[0] + "_tmean.nii.gz")
    mean_func_f = utils.tmean_img(ID=ID, i_img=func_file, o_img=o_img, verbose=False)
    ctrl_sc_file, mask_sc_file = preprocess_Sc.moco_mask(ID=ID,
                                                         i_img=mean_func_f,
                                                         mask_size_mm=35,
                                                         task_name=tag,
                                                         manual=manual_centerline,
                                                         redo_ctrl=redo,
                                                         redo_mask=redo,
                                                         verbose=verbose)

    print(mask_sc_file)
    print(f'=== Moco masks : Done  {ID} {tag} {run_name} ===', flush=True)

    # ------------------------------------------------------------------
    # ------ Run moco
    # ------------------------------------------------------------------
    moco_f, moco_mean_f, qc_dir = preprocess_Sc.moco(ID=ID,
                                                     i_img=func_file,
                                                     mask_img=mask_sc_file,
                                                     task_name=tag,
                                                     run_name=run_name,
                                                     params=params_moco,
                                                     verbose=verbose,
                                                     redo=redo,
                                                     use_dl=True)

    print(f'=== Moco : Done  {ID} {tag} {run_name} ===', flush=True)

    # ------------------------------------------------------------------
    # ------ Run func cord and CSF segmentation
    # ------------------------------------------------------------------
    # Cord segmentation
    seg_func_sc_file = preprocess_Sc.segmentation(ID=ID,
                                                  i_img=moco_mean_f,
                                                  task_name=tag,
                                                  img_type="func",
                                                  mask_qc=mask_sc_file,
                                                  redo=redo,
                                                  redo_qc=redo,  # should be true if you have done manual correction
                                                  verbose=verbose)
    # csf segmentation
    preprocess_Sc.segmentation(ID=ID,
                               i_img=moco_mean_f,
                               task_name=tag, contrast_anat="t2s",
                               img_type="func",
                               tissue="csf",
                               redo_qc=redo,  # should be true if you have done manual correction
                               redo=redo,
                               verbose=verbose)

    print(f'=== Func segmentation : Done  {ID} {tag} {run_name} ===', flush=True)

    # ------------------------------------------------------------------
    # ------ Registration in PAM50
    # ------------------------------------------------------------------
    param = "step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,metric=CC,iter=10,smooth=1,slicewise=1"
    func2PAM50_dir = preprocess_Sc.coreg_img2PAM50(ID=ID,
                                                   i_img=moco_mean_f,
                                                   i_seg=seg_func_sc_file,
                                                   task_name=tag,
                                                   run_name=run_name,
                                                   initwarp=warpT2w_PAM50_files[0],
                                                   initwarpinv=warpT2w_PAM50_files[1],
                                                   param=param,
                                                   redo=redo,
                                                   verbose=verbose)

    # Copy the segmentation, scf segmentation and warping field to where the final files are expected to be
    copy_segmentation_from_ref_tag(ID, tag, tag, manual_dir, preprocessing_dir)
    copy_csf_segmentation_from_ref_tag(ID, tag, tag, manual_dir, preprocessing_dir)
    copy_warping_fields_from_ref_tag(ID, tag, tag, preprocessing_dir)


def epi_avg_slices_moco(ID, source_tag, tag, n_slices_avg, redo, verbose):
    # ------------------------------------------------------------------
    # ------ Average n_slices_avg adjacent slices of the motion-corrected data
    # ------------------------------------------------------------------
    source_moco_dir = os.path.join(preprocessing_dir.format(ID), "func", source_tag, "sct_fmri_moco")
    moco_dir = os.path.join(preprocessing_dir.format(ID), "func", tag, "sct_fmri_moco")
    os.makedirs(moco_dir, exist_ok=True)

    source_moco_files = sorted(glob.glob(os.path.join(source_moco_dir, f"sub-{ID}_{source_tag}_*bold_moco.nii.gz")))

    outputs = []
    for source_moco_f in source_moco_files:
        match = re.search(r"_?(run-\d+)", source_moco_f)
        run_name = match.group(1) if match else ""
        run_tag = f"_{run_name}" if run_name else ""

        # De-jitter the per-slice AP offset (see issue #8) before averaging
        moco_params_y_f = os.path.join(source_moco_dir, f"moco_params_y_{source_tag}{run_tag}.nii.gz")
        destriped_f = os.path.join(moco_dir, os.path.basename(source_moco_f).replace(source_tag, tag).replace("_moco.nii.gz", "_moco_destriped.nii.gz"))
        destriped_f = utils.destripe_slices_img(i_img=source_moco_f, moco_params_img=moco_params_y_f, o_img=destriped_f, redo=redo, verbose=verbose)

        moco_f = os.path.join(moco_dir, os.path.basename(source_moco_f).replace(source_tag, tag))
        moco_f = utils.average_slices_img(i_img=destriped_f, o_img=moco_f, n_slices_avg=n_slices_avg, redo=redo, verbose=verbose)

        moco_mean_f = os.path.join(moco_dir, os.path.basename(moco_f).split(".")[0] + "_mean.nii.gz")
        moco_mean_f = utils.tmean_img(ID=ID, i_img=moco_f, o_img=moco_mean_f, redo=redo, verbose=verbose)

        outputs.append((moco_f, moco_mean_f, run_name))

    print(f'=== Slice averaging (1mm -> {n_slices_avg}x) : Done  {ID} {tag} ===', flush=True)

    return outputs


def epi_avg_slices_processing(ID, source_tag, tag, n_slices_avg, warpT2w_PAM50_files, redo, verbose):
    for moco_f, moco_mean_f, run_name in epi_avg_slices_moco(ID, source_tag, tag, n_slices_avg, redo, verbose):
        # ------------------------------------------------------------------
        # ------ Create mask around the cord for QC of the segmentation
        # ------------------------------------------------------------------
        ctrl_sc_file, mask_sc_file = preprocess_Sc.moco_mask(ID=ID,
                                                             i_img=moco_mean_f,
                                                             radius_size=25,
                                                             task_name=tag,
                                                             manual=manual_centerline,
                                                             redo_ctrl=redo,
                                                             redo_mask=redo,
                                                             verbose=verbose)

        # ------------------------------------------------------------------
        # ------ Run func cord and CSF segmentation
        # ------------------------------------------------------------------
        # Cord segmentation
        seg_func_sc_file = preprocess_Sc.segmentation(ID=ID,
                                                      i_img=moco_mean_f,
                                                      task_name=tag,
                                                      img_type="func",
                                                      mask_qc=mask_sc_file,
                                                      redo=redo,
                                                      redo_qc=redo,  # should be true if you have done manual correction
                                                      verbose=verbose)
        # csf segmentation
        preprocess_Sc.segmentation(ID=ID,
                                   i_img=moco_mean_f,
                                   task_name=tag, contrast_anat="t2s",
                                   img_type="func",
                                   tissue="csf",
                                   redo_qc=redo,  # should be true if you have done manual correction
                                   redo=redo,
                                   verbose=verbose)

        print(f'=== Func segmentation : Done  {ID} {tag} {run_name} ===', flush=True)

        # ------------------------------------------------------------------
        # ------ Registration in PAM50
        # ------------------------------------------------------------------
        param = "step=1,type=seg,algo=centermass:step=2,type=seg,algo=bsplinesyn,metric=CC,iter=10,smooth=1,slicewise=1"
        preprocess_Sc.coreg_img2PAM50(ID=ID,
                                       i_img=moco_mean_f,
                                       i_seg=seg_func_sc_file,
                                       task_name=tag,
                                       run_name=run_name,
                                       initwarp=warpT2w_PAM50_files[0],
                                       initwarpinv=warpT2w_PAM50_files[1],
                                       param=param,
                                       redo=redo,
                                       verbose=verbose)

        # Copy the segmentation, csf segmentation and warping field to where the final files are expected to be
        copy_segmentation_from_ref_tag(ID, tag, tag, manual_dir, preprocessing_dir)
        copy_csf_segmentation_from_ref_tag(ID, tag, tag, manual_dir, preprocessing_dir)
        copy_warping_fields_from_ref_tag(ID, tag, tag, preprocessing_dir)

#------------------------------------------------------------------
#------ Preprocessing
#------------------------------------------------------------------
print("=== Preprocessing script Start ===", flush=True)
print("Participant(s) included : ", IDs, flush=True)
print("===================================", flush=True)
print("")

for ID_nb, ID in enumerate(IDs):
    print("", flush=True)
    print(f'=== Preprocessing start for :  {ID} ===', flush=True)

    #---------------Anat preprocessing ---------------------------------------------------
    raw_anat = glob.glob(os.path.join(preprocessing_dir.format(ID), "anat", config["preprocess_f"]["anat_raw"].format(ID,"*")))[0]

    fname_anat_raw = glob.glob(os.path.join(config["raw_dir"], f"sub-{ID}", "anat", config["preprocess_f"]["anat_raw"].format(ID,"*")))[0]
    params = utils.extract_params(fname_anat_raw)
    params['run'] = ""
    params['ID'] = ID
    params['task'] = "anat"
    params['acq'] = "anat"
    acq_parameters.append(params)

    #------------------------------------------------------------------
    #------ Segmentation of the anatomical image
    #------------------------------------------------------------------

    seg_anat_sc_file = preprocess_Sc.segmentation(ID=ID,
                                                i_img=raw_anat,
                                                img_type="anat",
                                                contrast_anat="t2",
                                                redo=redo,
                                                redo_qc=redo, # should be true if you have done manual correction
                                                verbose=verbose)

    print(f'=== Anat segmentation : Done {ID} ===', flush=True)

    #------------------------------------------------------------------
    #------ Vertebral labelling
    #------------------------------------------------------------------
    # Manual fixing for participant
    # ID_using2labels = {"093": (3, 9)}

    disc_labels_files = preprocess_Sc.label_vertebrae(ID=ID,
                                                    i_img=raw_anat,
                                                    seg_img=seg_anat_sc_file,
                                                    c="t2",
                                                    auto=auto_vert_labels,
                                                    # labels_to_keep=ID_using2labels.get(ID),
                                                    redo=redo,
                                                    verbose=verbose)

    print(f'=== Anat vertebral labelling : Done {ID} ===', flush=True)

    #------------------------------------------------------------------
    #------ Registration in PAM50
    #------------------------------------------------------------------

    manual_seg_file = os.path.join(f"{manual_dir}", f"sub-{ID}", "anat", os.path.basename(seg_anat_sc_file))
    seg_anat_sc_final_file = manual_seg_file if os.path.exists(manual_seg_file) else seg_anat_sc_file
    param = "step=1,type=seg,algo=centermassrot"

    warpT2w_PAM50_files = preprocess_Sc.coreg_anat2PAM50(ID=ID,
                                                              i_img=raw_anat,
                                                              seg_img=seg_anat_sc_final_file,
                                                              labels_img=disc_labels_files,
                                                              img_type="t2",
                                                              tag='anat',
                                                              param=param,
                                                              redo=redo,
                                                              verbose=verbose)

    print(f'=== Registration anat to PAM50 : Done {ID} ===', flush=True)

    #---------------Func preprocessing ---------------------------------------------------
    #------ Select func data
    for task_name in config["design_exp"]["task_names"]:
        for acq_name in config["design_exp"]["acq_names"]:
            tag = "task-" + task_name + "_acq-" + acq_name
            raw_func = sorted(glob.glob(os.path.join(config["raw_dir"], f"sub-{ID}", "func", f"sub-{ID}_{tag}_*bold.nii.gz")))
            o_dir = os.path.join(preprocessing_dir.format(ID),  "func", tag)
            params_moco = 'poly=0,smooth=1,metric=MeanSquares,gradStep=1,sampling=0.2'

            if len(raw_func) == 0:
                print(f'No functional file found for {tag} in raw data, skipping this acquisition.', flush=True)
                os.rmdir(o_dir)
                continue

            for i_func, func_file in enumerate(raw_func):
                # Check run number if multiple runs exist
                match = re.search(r"_?(run-\d+)", func_file)
                if match:
                    run_name=match.group(1)
                    print(run_name)
                else:
                    run_name = ""

                params = utils.extract_params(func_file)
                params['run'] = run_name
                params['ID'] = ID
                params['task'] = task_name
                params['acq'] = acq_name
                acq_parameters.append(params)

                # Full processing only for the first motor task, the others will be co-registered
                if task_name == 'motor' and i_func == 0:
                    epi_full_processing(ID, func_file, tag, manual_centerline, warpT2w_PAM50_files, params_moco, o_dir, redo, verbose)
                else:
                    # For other tasks (eg: rest), we will use the motor fMRI scan as a reference for creating the moco mask, 
                    # as a target for moco, and for segmentation and registration to PAM50.
                    # See discussion: https://github.com/CarolineLndl/spine_7t_fmri_analysis/issues/64
                    # See discussion: https://github.com/CarolineLndl/spine_7t_fmri_analysis/issues/82
                    ref_tag = "task-motor_acq-" + acq_name
                    # reference func file has _tmean as a suffix, and is located under the ref_tag folder. If there are multiple runs, select the first one
                    try:
                        ref_func_file = glob.glob(os.path.join(preprocessing_dir.format(ID), "func", ref_tag, f"sub-{ID}_{ref_tag}_*bold_tmean.nii.gz"))[0]
                        print(f'=== Using {ref_func_file} as reference for moco ===', flush=True)
                    except IndexError as e:
                        print(f'No reference file found for {ref_tag} in raw data.', flush=True)
                        epi_full_processing(ID, func_file, tag, manual_centerline, warpT2w_PAM50_files, params_moco, o_dir, redo, verbose)
                        continue
                    # Reference mask file is located under ref_tag/sct_get_centerline. If there are multiple runs, select the first one
                    try:
                        ref_mask_file = glob.glob(os.path.join(preprocessing_dir.format(ID), "func", ref_tag, "sct_get_centerline", f"sub-{ID}_{ref_tag}_*tmean_mask.nii.gz"))[0]
                        print(f'=== Using {ref_mask_file} as reference mask for moco ===', flush=True)
                    except IndexError as e:
                        print(f'No reference mask file found for {ref_tag} in raw data.', flush=True)
                        raise e

                    # Run moco using the motor scan as reference, and using the same mask as for the motor scan
                    moco_f,moco_mean_f,qc_dir=preprocess_Sc.moco(ID=ID,
                                                                i_img=func_file,
                                                                mask_img=ref_mask_file,
                                                                ref_img=ref_func_file,
                                                                task_name=tag,
                                                                run_name=run_name,
                                                                params=params_moco,
                                                                verbose=verbose,
                                                                redo=redo,
                                                                use_dl=True)

                    # Copy segmentation and registration files from the motor task to the other tasks, since we are using
                    # the same reference for moco and registration to PAM50. This is to avoid having to redo segmentation
                    # and registration for the other tasks, which can be time-consuming.
                    copy_segmentation_from_ref_tag(ID, tag, ref_tag, manual_dir, preprocessing_dir)
                    copy_csf_segmentation_from_ref_tag(ID, tag, ref_tag, manual_dir, preprocessing_dir)
                    copy_warping_fields_from_ref_tag(ID, tag, ref_tag, preprocessing_dir)

                print(f'=== Func registration : Done  {ID} {tag} {run_name} ===')

    #---------------Derived (slice-averaged) acquisitions ---------------------------------
    for derived_acq_name, derived_info in config.get("derived_acq", {}).items():
        source_acq = derived_info["source_acq"]
        n_slices_avg = derived_info["n_slices_avg"]

        for task_name in config["design_exp"]["task_names"]:
            source_tag = "task-" + task_name + "_acq-" + source_acq
            tag = "task-" + task_name + "_acq-" + derived_acq_name

            source_moco_dir = os.path.join(preprocessing_dir.format(ID), "func", source_tag, "sct_fmri_moco")
            if not glob.glob(os.path.join(source_moco_dir, f"sub-{ID}_{source_tag}_*bold_moco.nii.gz")):
                print(f'No moco output found for {source_tag}, skipping derived acquisition {tag}.', flush=True)
                continue

            os.makedirs(os.path.join(preprocessing_dir.format(ID), "func", tag), exist_ok=True)

            if task_name == 'motor':
                epi_avg_slices_processing(ID, source_tag, tag, n_slices_avg, warpT2w_PAM50_files, redo, verbose)
            else:
                # For other tasks (eg: rest), co-register to PAM50 using the motor task's derived
                # acquisition as a reference, following the same pattern as the native acquisitions.
                ref_tag = "task-motor_acq-" + derived_acq_name
                ref_seg = os.path.join(preprocessing_dir.format(ID), "func", ref_tag, f"sub-{ID}_{ref_tag}_bold_moco_mean_seg.nii.gz")
                if not os.path.exists(ref_seg):
                    print(f'No reference outputs found for {ref_tag}, skipping derived acquisition {tag}.', flush=True)
                    continue
                epi_avg_slices_moco(ID, source_tag, tag, n_slices_avg, redo, verbose)
                copy_segmentation_from_ref_tag(ID, tag, ref_tag, manual_dir, preprocessing_dir)
                copy_csf_segmentation_from_ref_tag(ID, tag, ref_tag, manual_dir, preprocessing_dir)
                copy_warping_fields_from_ref_tag(ID, tag, ref_tag, preprocessing_dir)

            print(f'=== Slice-averaged processing : Done  {ID} {tag} ===', flush=True)

    print(f'=== Preprocessing done for : {ID} ===', flush=True)
    print("=========================================", flush=True)

df = pd.DataFrame(acq_parameters)
df_ordered = df[['ID', 'task', 'acq', 'run', 'EchoTime', 'RepetitionTime', 'FlipAngle', 'SliceThickness', 'SpacingBetweenSlices', 'NumberOfVolumes', 'BaseResolution', 'PartialFourier', 'ParallelReductionFactorInPlane', 'MultibandAccelerationFactor']]
df_ordered.to_csv(os.path.join(derivatives_dir, "processing", "acquisition_parameters.csv"), index=False)

# Print participant metrics
print(f"Number of particiants that have rest scans: {df_ordered[df_ordered['task'] == 'rest']['ID'].nunique()}", flush=True)

