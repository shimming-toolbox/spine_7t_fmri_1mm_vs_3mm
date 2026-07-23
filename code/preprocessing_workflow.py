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
import sys, json, glob, os, re, shutil, argparse, tempfile
import numpy as np
import nibabel as nib
import pandas as pd

# get path of the parent location of this file, and go up one level
path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(path_code, "code"))  # Change this line according to your directory
from preprocess import Preprocess_main, Preprocess_Sc, copy_warping_fields_from_ref_tag, copy_segmentation_from_ref_tag, manual_label_filename
import postprocess
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
manual_dir = os.path.join(config["raw_dir"], config["manual_dir"])



def destripe_if_sms(ID, tag, moco_f, moco_mean_f, redo, verbose):
    """
    For SMS acquisitions (identified by "sms" in tag), correct the even/odd slice
    AP jitter caused by the SMS slice-ordering scheme, then swap the destriped 4D
    volume into the canonical "*_moco.nii.gz" path -- the original raw sct_fmri_moco
    output is kept alongside, renamed "*_moco_not-destriped.nii.gz" -- and recompute
    the moco mean from it. No-op (returns moco_mean_f unchanged) otherwise.

    Swapping the destriped volume into the canonical path, rather than keeping it
    under a separate "_destriped" name, means every downstream consumer that globs
    for "*_moco.nii.gz" (tSNR, denoising, first-level GLM) automatically gets
    destriped data too, with no changes needed outside this function (#68).

    Used for both REST and MOTOR acquisitions: MOTOR is registered to REST
    (see epi_derive_seg_from_rest), so MOTOR must be destriped too, or that
    registration is degraded by residual jitter only present on one side.
    """
    if "sms" not in tag.lower():
        return moco_mean_f

    moco_dir = os.path.dirname(moco_f)
    run_tag = f"_{run_name}" if run_name else ""
    moco_params_y_f = os.path.join(moco_dir, f"moco_params_y_{tag}{run_tag}.nii.gz")
    if not os.path.exists(moco_params_y_f):
        print(f'WARNING: moco_params_y not found for {tag}, skipping destripe.', flush=True)
        return moco_mean_f

    not_destriped_f = moco_f.replace("_moco.nii.gz", "_moco_not-destriped.nii.gz")
    if not os.path.exists(not_destriped_f) or redo:
        destriped_tmp_f = moco_f.replace("_moco.nii.gz", "_moco_destriped_tmp.nii.gz")
        utils.destripe_slices_img(i_img=moco_f, moco_params_img=moco_params_y_f,
                                   o_img=destriped_tmp_f, redo=True, verbose=verbose)
        if os.path.exists(not_destriped_f):
            os.remove(not_destriped_f)
        os.rename(moco_f, not_destriped_f)
        os.rename(destriped_tmp_f, moco_f)
        moco_mean_f = utils.tmean_img(ID=ID, i_img=moco_f, o_img=moco_mean_f, redo=True, verbose=verbose)
        print(f'=== Destripe : Done  {ID} {tag} {run_name} ===', flush=True)
    return moco_mean_f


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
    # ------ Destripe (correct even/odd slice AP jitter from SMS)
    # ------------------------------------------------------------------
    moco_mean_f = destripe_if_sms(ID=ID, tag=tag, moco_f=moco_f, moco_mean_f=moco_mean_f, redo=redo, verbose=verbose)

    # ------------------------------------------------------------------
    # ------ Run func cord segmentation
    # ------------------------------------------------------------------
    seg_func_sc_file = preprocess_Sc.segmentation(ID=ID,
                                                  i_img=moco_mean_f,
                                                  task_name=tag,
                                                  img_type="func",
                                                  mask_qc=mask_sc_file,
                                                  redo=redo,
                                                  redo_qc=redo,  # should be true if you have done manual correction
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

    # Copy the segmentation and warping field to where the final files are expected to be
    copy_segmentation_from_ref_tag(ID, tag, tag, manual_dir, preprocessing_dir)
    copy_warping_fields_from_ref_tag(ID, tag, tag, preprocessing_dir)


def epi_derive_seg_from_rest(ID, rest_tag, func_file, tag, params_moco, o_dir, redo, verbose):
    """Moco MOTOR, register REST mean-moco -> MOTOR mean-moco, warp REST seg to MOTOR space."""
    # Moco for MOTOR using its own mean as reference
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

    # Destripe (correct even/odd slice AP jitter from SMS). MOTOR is registered to REST
    # below, so it must be destriped too, same as REST, or the registration is degraded
    # by residual jitter only present on the MOTOR side.
    moco_mean_f = destripe_if_sms(ID=ID, tag=tag, moco_f=moco_f, moco_mean_f=moco_mean_f, redo=redo, verbose=verbose)

    # Locate REST mean-moco and segmentations
    rest_moco_mean_candidates = sorted(glob.glob(os.path.join(
        preprocessing_dir.format(ID), "func", rest_tag, "sct_fmri_moco",
        f"sub-{ID}_{rest_tag}_*bold_moco_mean.nii.gz")))
    if not rest_moco_mean_candidates:
        raise RuntimeError(f"REST moco mean not found for {rest_tag} sub-{ID}; cannot derive MOTOR segmentation.")
    rest_moco_mean = rest_moco_mean_candidates[0]
    rest_sc_seg  = os.path.join(preprocessing_dir.format(ID), "func", rest_tag,
                                f"sub-{ID}_{rest_tag}_bold_moco_mean_seg.nii.gz")

    # Register REST mean-moco (moving) -> MOTOR mean-moco (fixed)
    reg_dir = os.path.join(o_dir, "sct_register_rest2motor")
    os.makedirs(reg_dir, exist_ok=True)
    warp_rest2motor  = os.path.join(reg_dir, f"sub-{ID}_{tag}_from-rest_to-motor_xfm.nii.gz")
    warp_motor2rest  = os.path.join(reg_dir, f"sub-{ID}_{tag}_from-motor_to-rest_xfm.nii.gz")
    if not os.path.exists(warp_rest2motor) or not os.path.exists(warp_motor2rest) or redo:
        cmd_reg = (f"sct_register_multimodal -i {rest_moco_mean} -d {moco_mean_f}"
                   f" -dseg {mask_sc_file}"
                   f" -param step=1,type=im,algo=affine,metric=CC -ofolder {reg_dir}"
                   f" -qc {preprocess_Sc.qc_dir} -qc-subject sub-{ID} -qc-contrast {tag} -v 0")
        os.system(cmd_reg)
        # sct_register_multimodal names warps after src/dest basenames
        src_stem  = os.path.basename(rest_moco_mean).replace(".nii.gz", "")
        dest_stem = os.path.basename(moco_mean_f).replace(".nii.gz", "")
        fwd = glob.glob(os.path.join(reg_dir, f"warp_{src_stem}2*.nii.gz"))
        inv = glob.glob(os.path.join(reg_dir, f"warp_{dest_stem}2*.nii.gz"))
        if not fwd or not inv:
            raise RuntimeError(f"REST<->MOTOR registration warps not found in {reg_dir}")
        os.rename(fwd[0], warp_rest2motor)
        os.rename(inv[0], warp_motor2rest)

    # Apply warp to REST SC segmentation -> MOTOR space
    motor_sc_seg = os.path.join(o_dir, f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz")
    if not os.path.exists(motor_sc_seg) or redo:
        cmd_apply = (f"sct_apply_transfo -i {rest_sc_seg} -d {moco_mean_f}"
                     f" -w {warp_rest2motor} -o {motor_sc_seg} -x nn -v 0")
        os.system(cmd_apply)
    print(f'=== Derived seg from REST: Done  {ID} {tag} {run_name} ===', flush=True)

    # Compose PAM50<->MOTOR warp files from REST's PAM50 warp + REST<->MOTOR registration.
    # No new sct_register_multimodal call needed: the REST registration is valid for MOTOR
    # since they share the same FOV, and any residual offset is captured by sct_register_rest2motor.
    rest_to_pam50 = os.path.join(preprocessing_dir.format(ID), "func", rest_tag,
                                  f"sub-{ID}_{rest_tag}_from-func_to_PAM50_mode-image_xfm.nii.gz")
    pam50_to_rest = os.path.join(preprocessing_dir.format(ID), "func", rest_tag,
                                  f"sub-{ID}_{rest_tag}_from-PAM50_to_func_mode-image_xfm.nii.gz")
    pam50_t2 = os.path.join(preprocess_Sc.code_dir, "template", preprocess_Sc.config["PAM50_t2"])

    run_tag_str = f"_{run_name}" if run_name else ""
    func2pam50_dir = os.path.join(o_dir, "sct_register_multimodal")
    os.makedirs(func2pam50_dir, exist_ok=True)
    motor_to_pam50 = os.path.join(func2pam50_dir,
                                   f"sub-{ID}_{tag}{run_tag_str}_from-func_to_PAM50_mode-image_xfm.nii.gz")
    pam50_to_motor = os.path.join(func2pam50_dir,
                                   f"sub-{ID}_{tag}{run_tag_str}_from-PAM50_to_func_mode-image_xfm.nii.gz")

    # isct_ComposeMultiTransform applies transforms right-to-left (last arg applied first).
    # motor_to_pam50 (pull: PAM50->MOTOR, for stat map normalization in firstlevel):
    #   PAM50->REST (rest_to_pam50) then REST->MOTOR (warp_motor2rest)
    if not os.path.exists(motor_to_pam50) or redo:
        cmd = (f"isct_ComposeMultiTransform 3 {motor_to_pam50} -R {pam50_t2}"
               f" {warp_motor2rest} {rest_to_pam50}")
        os.system(cmd)

    # pam50_to_motor (pull: MOTOR->PAM50, for copy_warping_fields_from_ref_tag):
    #   MOTOR->REST (warp_rest2motor) then REST->PAM50 (pam50_to_rest)
    if not os.path.exists(pam50_to_motor) or redo:
        cmd = (f"isct_ComposeMultiTransform 3 {pam50_to_motor} -R {moco_mean_f}"
               f" {pam50_to_rest} {warp_rest2motor}")
        os.system(cmd)

    copy_warping_fields_from_ref_tag(ID, tag, tag, preprocessing_dir)

    # Generate QC: bring PAM50 T2 into MOTOR space by chaining pam50_to_rest then warp_rest2motor.
    # Use isct_antsApplyTransforms directly: sct_apply_transfo silently drops all but the last -w
    # when multiple displacement fields are passed. ANTs applies -t flags right-to-left (last first),
    # so listing pam50_to_rest before warp_rest2motor means warp_rest2motor is applied first
    # (MOTOR→REST), then pam50_to_rest (REST→PAM50) — correct pull-convention composition.
    pam50_t2_reg = os.path.join(func2pam50_dir, f"PAM50_t2_reg{run_tag_str}.nii.gz")
    if not os.path.exists(pam50_t2_reg) or redo:
        cmd = (f"isct_antsApplyTransforms -d 3 -i {pam50_t2} -o {pam50_t2_reg}"
               f" -t {pam50_to_rest} -t {warp_rest2motor}"
               f" -r {moco_mean_f} -n 'BSpline[3]'")
        os.system(cmd)
    cmd_qc = (f"sct_qc -i {moco_mean_f} -s {motor_sc_seg} -p sct_register_multimodal"
              f" -d {pam50_t2_reg} -qc {preprocess_Sc.qc_dir} -qc-subject sub-{ID}"
              f" -qc-contrast {tag} -v 0")
    os.system(cmd_qc)

    # Warp MOTOR moco mean into PAM50 space — used by MI computation in figures_workflow.
    moco_mean_base = os.path.basename(moco_mean_f).replace(".nii.gz", "")
    coreg_in_pam50 = os.path.join(func2pam50_dir, f"{moco_mean_base}_coreg_in_PAM50.nii.gz")
    if not os.path.exists(coreg_in_pam50) or redo:
        cmd = (f"sct_apply_transfo -i {moco_mean_f} -d {pam50_t2}"
               f" -w {motor_to_pam50} -o {coreg_in_pam50} -x spline -v 0")
        os.system(cmd)
    print(f'=== PAM50 registration (MOTOR, warp composition): Done  {ID} {tag} {run_name} ===', flush=True)


def _get_seg_file(ID, source_tag):
    """Return path to SC segmentation for source_tag, preferring manual over auto.

    Checks sct_deepseg subfolder first (REST, produced by epi_full_processing),
    then falls back to the tag root (MOTOR, produced by epi_derive_seg_from_rest).
    """
    tag_dir = os.path.join(preprocessing_dir.format(ID), "func", source_tag)
    manual_files = glob.glob(os.path.join(manual_dir, f"sub-{ID}", "func",
                                          f"sub-{ID}_{source_tag}_*bold_moco_mean_label-SC_seg.nii.gz"))
    auto_files = (glob.glob(os.path.join(tag_dir, "sct_deepseg",
                                         f"sub-{ID}_{source_tag}_*bold_moco_mean_seg.nii.gz")) or
                  glob.glob(os.path.join(tag_dir,
                                         f"sub-{ID}_{source_tag}_*bold_moco_mean_seg.nii.gz")))
    files = manual_files or auto_files
    if not files:
        raise RuntimeError(f"No segmentation found for {source_tag} sub-{ID}")
    return sorted(files)[0]


def epi_avg_slices_moco(ID, source_tag, tag, n_slices_avg, redo, verbose):
    # ------------------------------------------------------------------
    # ------ Average n_slices_avg adjacent slices of the source's temporal mean.
    # ------ This is mathematically equivalent to averaging slices of the full 4D
    # ------ series and then taking the temporal mean (both are linear averages
    # ------ over independent axes), but far cheaper, and avoids materializing a
    # ------ full 4D series that nothing downstream actually reads.
    # ------------------------------------------------------------------
    source_moco_dir = os.path.join(preprocessing_dir.format(ID), "func", source_tag, "sct_fmri_moco")
    moco_dir = os.path.join(preprocessing_dir.format(ID), "func", tag, "sct_fmri_moco")
    os.makedirs(moco_dir, exist_ok=True)

    source_mean_files = sorted(glob.glob(os.path.join(source_moco_dir, f"sub-{ID}_{source_tag}_*bold_moco_mean.nii.gz")))

    outputs = []
    for source_mean_f in source_mean_files:
        match = re.search(r"_?(run-\d+)", source_mean_f)
        run_name = match.group(1) if match else ""

        moco_mean_f = os.path.join(moco_dir, os.path.basename(source_mean_f).replace(source_tag, tag))
        moco_mean_f = utils.average_slices_img(i_img=source_mean_f, o_img=moco_mean_f, n_slices_avg=n_slices_avg, redo=redo, verbose=verbose)

        outputs.append((moco_mean_f, run_name))

    print(f'=== Slice averaging (1mm -> {n_slices_avg}x) : Done  {ID} {tag} ===', flush=True)

    return outputs


def epi_avg_slices_processing(ID, source_tag, tag, n_slices_avg, redo, verbose):
    for moco_mean_f, run_name in epi_avg_slices_moco(ID, source_tag, tag, n_slices_avg, redo, verbose):
        seg_func_sc_file = os.path.join(preprocessing_dir.format(ID), "func", tag,
                                        f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz")
        if not os.path.exists(seg_func_sc_file) or redo:
            src = _get_seg_file(ID, source_tag)
            # Resample (not block-average) the source segmentation onto the averaged-slice
            # grid: sct_register_multimodal -identity 1 does a pure resampling (no
            # registration search), which avoids the empty edge slices that a naive
            # slice-block-average + threshold could produce.
            with tempfile.TemporaryDirectory() as tmpdir:
                cmd = (f"sct_register_multimodal -i {src} -d {moco_mean_f} "
                       f"-identity 1 -x linear -ofolder {tmpdir} -v 0")
                os.system(cmd)
                reg_f = os.path.join(tmpdir, os.path.basename(src).replace(".nii.gz", "_reg.nii.gz"))
                nii = nib.load(reg_f)
                binary = (nii.get_fdata() >= 0.5).astype(np.uint8)
                nib.save(nib.Nifti1Image(binary, nii.affine, nii.header), seg_func_sc_file)

        print(f'=== Derived seg (resampled from {source_tag}): Done {ID} {tag} {run_name} ===', flush=True)


def epi_smooth_slices_moco(ID, source_tag, tag, smooth_width, redo, verbose):
    # ------------------------------------------------------------------
    # ------ Apply sliding-window z-smoothing (keeps full slice count)
    # ------------------------------------------------------------------
    source_moco_dir = os.path.join(preprocessing_dir.format(ID), "func", source_tag, "sct_fmri_moco")
    moco_dir = os.path.join(preprocessing_dir.format(ID), "func", tag, "sct_fmri_moco")
    os.makedirs(moco_dir, exist_ok=True)

    # Prefer already-destriped source (produced by epi_full_processing for sms acqs)
    source_moco_files = (sorted(glob.glob(os.path.join(source_moco_dir, f"sub-{ID}_{source_tag}_*bold_moco_destriped.nii.gz"))) or
                         sorted(glob.glob(os.path.join(source_moco_dir, f"sub-{ID}_{source_tag}_*bold_moco.nii.gz"))))

    outputs = []
    for source_moco_f in source_moco_files:
        match = re.search(r"_?(run-\d+)", source_moco_f)
        run_name = match.group(1) if match else ""

        moco_f = os.path.join(moco_dir, os.path.basename(source_moco_f)
                              .replace(source_tag, tag)
                              .replace("_moco_destriped.nii.gz", "_moco.nii.gz"))
        moco_f = utils.smooth_slices_img(i_img=source_moco_f, o_img=moco_f, smooth_width=smooth_width, redo=redo, verbose=verbose)

        moco_mean_f = os.path.join(moco_dir, os.path.basename(moco_f).split(".")[0] + "_mean.nii.gz")
        moco_mean_f = utils.tmean_img(ID=ID, i_img=moco_f, o_img=moco_mean_f, redo=redo, verbose=verbose)

        outputs.append((moco_f, moco_mean_f, run_name))

    print(f'=== Slice smoothing (width={smooth_width}) : Done  {ID} {tag} ===', flush=True)

    return outputs


def epi_smooth_slices_processing(ID, source_tag, tag, smooth_width, redo, verbose):
    for moco_f, moco_mean_f, run_name in epi_smooth_slices_moco(ID, source_tag, tag, smooth_width, redo, verbose):
        seg_func_sc_file = os.path.join(preprocessing_dir.format(ID), "func", tag,
                                        f"sub-{ID}_{tag}_bold_moco_mean_seg.nii.gz")
        if not os.path.exists(seg_func_sc_file) or redo:
            shutil.copy(_get_seg_file(ID, source_tag), seg_func_sc_file)

        print(f'=== Derived seg (copy from {source_tag}): Done {ID} {tag} {run_name} ===', flush=True)

        # Copy PAM50 warp fields from the 1mm source: z-smoothing does not change geometry.
        copy_warping_fields_from_ref_tag(ID, tag, source_tag, preprocessing_dir)

        # Warp smooth3mm moco mean into PAM50 space — used by MI computation in figures_workflow.
        pam50_t2 = os.path.join(preprocess_Sc.code_dir, "template", preprocess_Sc.config["PAM50_t2"])
        func_to_pam50 = os.path.join(preprocessing_dir.format(ID), "func", tag,
                                     f"sub-{ID}_{tag}_from-func_to_PAM50_mode-image_xfm.nii.gz")
        coreg_dir = os.path.join(preprocessing_dir.format(ID), "func", tag, "sct_register_multimodal")
        os.makedirs(coreg_dir, exist_ok=True)
        moco_mean_base = os.path.basename(moco_mean_f).replace(".nii.gz", "")
        coreg_in_pam50 = os.path.join(coreg_dir, f"{moco_mean_base}_coreg_in_PAM50.nii.gz")
        if not os.path.exists(coreg_in_pam50) or redo:
            cmd = (f"sct_apply_transfo -i {moco_mean_f} -d {pam50_t2}"
                   f" -w {func_to_pam50} -o {coreg_in_pam50} -x spline -v 0")
            os.system(cmd)

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

    manual_seg_file = os.path.join(f"{manual_dir}", f"sub-{ID}", "anat", manual_label_filename(os.path.basename(seg_anat_sc_file), "SC"))
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
    # Sort REST before MOTOR so that MOTOR segmentation can be derived from REST.
    for task_name in sorted(config["design_exp"]["task_names"], key=lambda t: 0 if t == 'rest' else 1):
        for acq_name in config["design_exp"]["acq_names"]:
            tag = "task-" + task_name + "_acq-" + acq_name
            raw_func = sorted(glob.glob(os.path.join(config["raw_dir"], f"sub-{ID}", "func", f"sub-{ID}_{tag}_*bold.nii.gz")))
            o_dir = os.path.join(preprocessing_dir.format(ID),  "func", tag)
            params_moco = 'poly=0,smooth=1,metric=MeanSquares,gradStep=1,sampling=0.2'

            if len(raw_func) == 0:
                print(f'No functional file found for {tag} in raw data, skipping this acquisition.', flush=True)
                if os.path.isdir(o_dir):
                    try:
                        os.rmdir(o_dir)
                    except OSError:
                        pass
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

                if task_name == 'rest' and i_func == 0:
                    # REST: full processing (moco + sct_deepseg + PAM50 registration).
                    # REST mean-moco is cleaner (no task confounds), making it the better
                    # substrate for segmentation. MOTOR will derive its seg from this.
                    epi_full_processing(ID, func_file, tag, manual_centerline, warpT2w_PAM50_files, params_moco, o_dir, redo, verbose)

                elif task_name == 'motor' and i_func == 0:
                    # MOTOR: moco using own mean, then register REST mean-moco -> MOTOR mean-moco
                    # and warp the REST segmentation into MOTOR space.
                    rest_tag = "task-rest_acq-" + acq_name
                    rest_moco_mean_candidates = glob.glob(os.path.join(
                        preprocessing_dir.format(ID), "func", rest_tag, "sct_fmri_moco",
                        f"sub-{ID}_{rest_tag}_*bold_moco_mean.nii.gz"))
                    if rest_moco_mean_candidates:
                        epi_derive_seg_from_rest(ID, rest_tag, func_file, tag, params_moco, o_dir, redo, verbose)
                    else:
                        # REST not available for this acq (e.g. sub-099 1mm) — fall back to full processing.
                        print(f'No REST moco mean found for {rest_tag}; running full processing for MOTOR.', flush=True)
                        epi_full_processing(ID, func_file, tag, manual_centerline, warpT2w_PAM50_files, params_moco, o_dir, redo, verbose)

                else:
                    # Additional runs (i_func > 0): run moco referencing own task's first run,
                    # then copy segmentation and warp fields from that first run.
                    ref_tag = "task-" + task_name + "_acq-" + acq_name
                    try:
                        ref_func_file = glob.glob(os.path.join(preprocessing_dir.format(ID), "func", ref_tag, f"sub-{ID}_{ref_tag}_*bold_tmean.nii.gz"))[0]
                        print(f'=== Using {ref_func_file} as reference for moco ===', flush=True)
                    except IndexError as e:
                        print(f'No reference file found for {ref_tag} in raw data.', flush=True)
                        epi_full_processing(ID, func_file, tag, manual_centerline, warpT2w_PAM50_files, params_moco, o_dir, redo, verbose)
                        continue
                    try:
                        ref_mask_file = glob.glob(os.path.join(preprocessing_dir.format(ID), "func", ref_tag, "sct_get_centerline", f"sub-{ID}_{ref_tag}_*tmean_mask.nii.gz"))[0]
                        print(f'=== Using {ref_mask_file} as reference mask for moco ===', flush=True)
                    except IndexError as e:
                        print(f'No reference mask file found for {ref_tag} in raw data.', flush=True)
                        raise e

                    moco_f, moco_mean_f, qc_dir = preprocess_Sc.moco(ID=ID,
                                                                i_img=func_file,
                                                                mask_img=ref_mask_file,
                                                                ref_img=ref_func_file,
                                                                task_name=tag,
                                                                run_name=run_name,
                                                                params=params_moco,
                                                                verbose=verbose,
                                                                redo=redo,
                                                                use_dl=True)

                    copy_segmentation_from_ref_tag(ID, tag, ref_tag, manual_dir, preprocessing_dir)
                    copy_warping_fields_from_ref_tag(ID, tag, ref_tag, preprocessing_dir)

                print(f'=== Func registration : Done  {ID} {tag} {run_name} ===')

    #---------------Derived acquisitions (slice-averaged or z-smoothed) -------------------
    # Sort REST before MOTOR so MOTOR-derived can use the MOTOR source seg
    # (which was itself derived from REST via registration in the loop above).
    for derived_acq_name, derived_info in config.get("derived_acq", {}).items():
        source_acq = derived_info["source_acq"]
        n_slices_avg = derived_info.get("n_slices_avg")
        smooth_width = derived_info.get("smooth_width")

        for task_name in sorted(config["design_exp"]["task_names"], key=lambda t: 0 if t == 'rest' else 1):
            source_tag = "task-" + task_name + "_acq-" + source_acq
            tag = "task-" + task_name + "_acq-" + derived_acq_name

            source_moco_dir = os.path.join(preprocessing_dir.format(ID), "func", source_tag, "sct_fmri_moco")
            if not glob.glob(os.path.join(source_moco_dir, f"sub-{ID}_{source_tag}_*bold_moco.nii.gz")):
                print(f'No moco output found for {source_tag}, skipping derived acquisition {tag}.', flush=True)
                continue

            os.makedirs(os.path.join(preprocessing_dir.format(ID), "func", tag), exist_ok=True)

            if n_slices_avg is not None:
                epi_avg_slices_processing(ID, source_tag, tag, n_slices_avg, redo, verbose)
            else:
                epi_smooth_slices_processing(ID, source_tag, tag, smooth_width, redo, verbose)

            print(f'=== Derived acquisition processing : Done  {ID} {tag} ===', flush=True)

    print(f'=== Preprocessing done for : {ID} ===', flush=True)
    print("=========================================", flush=True)

df = pd.DataFrame(acq_parameters)
df_ordered = df[['ID', 'task', 'acq', 'run', 'EchoTime', 'RepetitionTime', 'FlipAngle', 'SliceThickness', 'SpacingBetweenSlices', 'NumberOfVolumes', 'BaseResolution', 'PartialFourier', 'ParallelReductionFactorInPlane', 'MultibandAccelerationFactor']]
df_ordered.to_csv(os.path.join(preprocessing_dir.format("").split("sub")[0], "acquisition_parameters.csv"), index=False)

# Print participant metrics
print(f"Number of particiants that have rest scans: {df_ordered[df_ordered['task'] == 'rest']['ID'].nunique()}", flush=True)

# ------------------------------------------------------------------
# Compute tSNR maps (REST data only)
# ------------------------------------------------------------------
print("", flush=True)
print("=== tSNR script Start ===", flush=True)
config_tsnr = dict(config)
config_tsnr["design_exp"] = dict(config["design_exp"])
config_tsnr["design_exp"]["task_names"] = ["rest"]
tsnr_ana = postprocess.TSNR_main(config_tsnr, IDs, redo)
tsnr_ana.generate_tsnr_maps_and_csv()
tsnr_ana.generate_tsnr_maps_derived()
print("=== tSNR script Done ===", flush=True)

