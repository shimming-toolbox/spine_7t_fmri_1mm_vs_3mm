"""
Motion correction with the new moco-dl model (ivadomed/moco-dl#25), until it is integrated into SCT.

Reproduces `infer.py` from the moco-dl `td/25-inference` branch (two passes: model 1 with
nearest-neighbour warping, then model 2 with bilinear warping on the pass-1 output), and in
addition writes per-slice, per-volume translations in the same format as `sct_fmri_moco -dl`
(`moco_params_x.nii.gz`, `moco_params_y.nii.gz`: shape (1, 1, n_slices, n_vols), in mm), which
this pipeline needs for destriping, framewise displacement and the moco plots.

This script must run in the moco-dl Python environment (torch, monai, sc_crop), not in SCT's:
    $MOCO_DL_PYTHON code/moco_dl_v2.py -i bold.nii.gz -o bold_moco.nii.gz -ofolder <dir>
with $MOCO_DL_DIR pointing to the moco-dl clone (containing infer.py and checkpoints/).
"""
import argparse
import os
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
import torch

MOCO_DL_DIR = os.path.expandvars(os.environ.get("MOCO_DL_DIR"))
if not MOCO_DL_DIR or not os.path.exists(os.path.join(MOCO_DL_DIR, "infer.py")):
    sys.exit("MOCO_DL_DIR must point to a clone of ivadomed/moco-dl (td/25-inference) containing infer.py")
sys.path.insert(0, MOCO_DL_DIR)
# infer.py calls the `sc_crop` CLI, installed next to this interpreter (no need to activate the venv)
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")

import infer  # noqa: E402  (moco-dl)
from coord_transform import pixel_affine_from_theta  # noqa: E402  (moco-dl)


def run_pass(in_path, checkpoint, out_path, crop_path, interpolation):
    """
    One inference pass, identical to infer.run_inference(reference="average"), but also returns
    the transform applied to each raw slice as a pixel affine (pixel_in = A @ pixel_out + b,
    (x, y) = (array axis 1, array axis 0)), and the crop centre in raw pixels.
    """
    infer.run_sc_crop(in_path, crop_path)
    raw_img = nib.load(in_path)
    raw_data = raw_img.get_fdata(dtype=np.float32)
    cropped_img = nib.load(crop_path)
    cropped_data = cropped_img.get_fdata(dtype=np.float32)
    X_raw, Y_raw, Z_raw, T_raw = raw_data.shape
    Xc, Yc, Zc, _ = cropped_data.shape

    ox, oy, oz = infer.get_crop_offset_voxels(raw_img, cropped_img)
    crop_center_in_raw_px = (oy + Yc / 2.0, ox + Xc / 2.0)  # (x, y), see infer.py

    model = infer.load_checkpoint(infer.build_model(), checkpoint)
    reference = infer.compute_reference(cropped_data, "average")

    # identity for slices outside the crop's z-range (passed through uncorrected, as in infer.py)
    A_all = np.tile(np.eye(2), (Z_raw, T_raw, 1, 1))
    b_all = np.zeros((Z_raw, T_raw, 2))
    corrected = raw_data.copy()

    for t in range(T_raw):
        moving, fixed = [], []
        for z in range(Zc):
            moving.append(infer.to_model_space(infer.normalize(cropped_data[:, :, z, t]))[0])
            fixed.append(infer.to_model_space(infer.normalize(reference[:, :, z]))[0])
        moving_t = torch.from_numpy(np.stack(moving)[:, None]).float()
        fixed_t = torch.from_numpy(np.stack(fixed)[:, None]).float()
        with torch.no_grad():
            _, affine_matrix = model(moving_t, fixed_t)
        affine_matrix = affine_matrix.cpu().numpy()

        for z in range(Zc):
            raw_z = z + oz
            if raw_z < 0 or raw_z >= Z_raw:
                continue
            theta_raw = infer.transfer_theta_to_raw(affine_matrix[z], infer.TARGET_SHAPE,
                                                    crop_center_in_raw_px, (X_raw, Y_raw))
            A_all[raw_z, t], b_all[raw_z, t] = pixel_affine_from_theta(theta_raw, (X_raw, Y_raw))
            raw_slice = torch.from_numpy(raw_data[:, :, raw_z, t])[None, None].float()
            warped = infer.warp_with_affine(raw_slice, torch.from_numpy(theta_raw)[None].float(), mode=interpolation)
            corrected[:, :, raw_z, t] = warped[0, 0].numpy()

    nib.save(nib.Nifti1Image(corrected.astype(np.float32), raw_img.affine, raw_img.header), out_path)
    return A_all, b_all, np.array(crop_center_in_raw_px)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", required=True, help="Raw 4D fMRI image")
    parser.add_argument("-o", required=True, help="Motion-corrected 4D output")
    parser.add_argument("-ofolder", required=True, help="Folder for moco_params_x/y.nii.gz")
    parser.add_argument("-ocropbox", help="Optional: save sc_crop's box around the cord (found on the raw image) here")
    args = parser.parse_args()

    checkpoints = os.path.join(MOCO_DL_DIR, "checkpoints")
    with tempfile.TemporaryDirectory() as tmp:
        # sc_crop writes *_bbox.txt and *_cropbox.nii.gz next to its input: work on a copy so
        # nothing is written into the raw dataset
        raw = os.path.join(tmp, "raw.nii.gz")
        shutil.copy(args.i, raw)
        pass1 = os.path.join(tmp, "pass1.nii.gz")
        A1, b1, center = run_pass(raw, os.path.join(checkpoints, "first_model.pt"), pass1,
                                  os.path.join(tmp, "crop1.nii.gz"), "nearest")
        A2, b2, _ = run_pass(pass1, os.path.join(checkpoints, "second_model.pt"), args.o,
                             os.path.join(tmp, "crop2.nii.gz"), "bilinear")
        if args.ocropbox:
            shutil.copy(os.path.join(tmp, "raw_cropbox.nii.gz"), args.ocropbox)

    # Compose both passes: out(p) = pass1(A2 p + b2) = raw(A1 (A2 p + b2) + b1).
    # Motion parameter = displacement of the cord (crop centre) between output and raw, i.e. where
    # the output samples the raw image from, minus where it is: (A c + b) - c.
    A = A1 @ A2
    b = np.einsum("ztij,ztj->zti", A1, b2) + b1
    disp_px = np.einsum("ztij,j->zti", A, center) + b - center  # (Z, T, (x, y)) in pixels
    zooms = nib.load(args.i).header.get_zooms()[:3]
    hdr = nib.load(args.i).header.copy()
    os.makedirs(args.ofolder, exist_ok=True)
    # x = array axis 0, y = array axis 1, as in sct_fmri_moco -dl outputs
    for name, d in (("x", disp_px[..., 1] * zooms[0]), ("y", -disp_px[..., 0] * zooms[1])):
        hdr.set_data_shape((1, 1) + d.shape)
        hdr.set_data_dtype(np.float32)
        nib.save(nib.Nifti1Image(d[None, None].astype(np.float32), nib.load(args.i).affine, hdr),
                 os.path.join(args.ofolder, f"moco_params_{name}.nii.gz"))


if __name__ == "__main__":
    main()
