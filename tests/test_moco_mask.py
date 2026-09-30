# -*- coding: utf-8 -*-
"""Tests for `Preprocess_Sc.moco_mask()`.

Regression coverage for issue #113: a manually drawn centerline under
`derivatives/manual/` must be the one the moco mask is built from. Before the fix the
manual file was only looked up *after* `sct_create_mask` had already run on the
automatic centerline, so manual corrections never reached motion correction.

The SCT command-line calls are stubbed out, so these tests need no data and no SCT
binaries -- but `code/preprocess.py` imports nibabel/pandas/matplotlib at module level,
so run them with the pipeline's Python:

    $SCT_DIR/python/envs/venv_sct/bin/python3 -m pytest tests/ -v
"""
import os
import sys

import nibabel as nib
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))

import preprocess  # noqa: E402

TAG = "task-motor_acq-shimSlice+3mm"


def _write_nii(path, value=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nib.save(nib.Nifti1Image(np.full((4, 4, 3), value, dtype=np.uint8), np.eye(4)), path)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A minimal BIDS-ish tree plus a stubbed `os.system` that records SCT calls.

    Returns (instance, paths, calls) where `calls` accumulates every command issued.
    """
    root = str(tmp_path)
    config = {
        "raw_dir": root,
        "derivatives_dir": "derivatives",
        "manual_dir": os.path.join("derivatives", "manual"),
        "code_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"),
        "structures": ["spinalcord"],
        "preprocess_dir": {
            "QC_dir": "QC",
            "main_dir": os.path.join("derivatives", "preprocessing", "sub-{}"),
            "func_mask": os.path.join("func", "{}", "sct_get_centerline"),
        },
    }
    instance = preprocess.Preprocess_Sc(config, IDs=["100"])
    # `main_dir` goes through `str.format(ID)`, so keep the placeholder unresolved here.
    instance.preprocessing_dir = os.path.join(root, "derivatives", "preprocessing", "sub-{}")

    func_dir = os.path.join(root, "derivatives", "preprocessing", "sub-100", "func", TAG)
    # `self.structure` is "" for a single-structure config (as in config_spine_7t_fmri.json),
    # so derive it rather than hard-coding a "spinalcord" subfolder that would not exist.
    ctl_dir = os.path.join(func_dir, "sct_get_centerline", instance.structure)
    paths = {
        "i_img": os.path.join(func_dir, f"sub-100_{TAG}_bold_tmean.nii.gz"),
        "auto_centerline": os.path.join(ctl_dir, f"sub-100_{TAG}_bold_tmean_centerline.nii.gz"),
        "manual_centerline": os.path.join(root, "derivatives", "manual", "sub-100", "func",
                                          instance.structure,
                                          f"sub-100_{TAG}_bold_tmean_centerline.nii.gz"),
        "mask": os.path.join(ctl_dir, f"sub-100_{TAG}_bold_tmean_mask.nii.gz"),
    }
    _write_nii(paths["i_img"])

    calls = []

    def fake_system(cmd):
        """Record the command and fabricate whatever output file it declares via -o."""
        calls.append(cmd)
        if cmd.startswith("sct_get_centerline"):
            _write_nii(cmd.split(" -o ")[1].split(" ")[0] + ".nii.gz")
        elif cmd.startswith("sct_create_mask"):
            _write_nii(cmd.split(" -o ")[1].split(" ")[0], value=1)
        return 0

    monkeypatch.setattr(preprocess.os, "system", fake_system)
    return instance, paths, calls


def _centerline_feeding_mask(calls):
    """The centerline path passed to `sct_create_mask -p centerline,<path>`."""
    mask_cmds = [c for c in calls if c.startswith("sct_create_mask")]
    assert mask_cmds, "sct_create_mask was never called"
    return mask_cmds[0].split("-p centerline,")[1].split(" ")[0]


def test_uses_automatic_centerline_when_no_manual_exists(sandbox):
    instance, paths, calls = sandbox

    instance.moco_mask(ID="100", i_img=paths["i_img"], task_name=TAG, verbose=False)

    assert _centerline_feeding_mask(calls) == paths["auto_centerline"]


def test_manual_centerline_takes_precedence(sandbox):
    """Issue #113: the manual centerline must be what the mask is built from."""
    instance, paths, calls = sandbox
    _write_nii(paths["manual_centerline"])

    instance.moco_mask(ID="100", i_img=paths["i_img"], task_name=TAG, verbose=False)

    assert _centerline_feeding_mask(calls) == paths["manual_centerline"]


def test_stale_mask_is_rebuilt_from_manual_centerline(sandbox):
    """A mask left over from an earlier automatic run must not be silently reused.

    `sct_create_mask` is normally skipped when the mask already exists, and mtime is no
    help here: the manual centerline is typically *older* than the mask built from the
    automatic one. So a manual centerline has to force the rebuild.
    """
    instance, paths, calls = sandbox
    _write_nii(paths["manual_centerline"])
    _write_nii(paths["mask"], value=1)  # as if built from the automatic centerline earlier

    instance.moco_mask(ID="100", i_img=paths["i_img"], task_name=TAG, verbose=False)

    assert _centerline_feeding_mask(calls) == paths["manual_centerline"]


def test_returns_the_centerline_actually_used(sandbox):
    instance, paths, calls = sandbox
    _write_nii(paths["manual_centerline"])

    centerline_f, mask_f = instance.moco_mask(ID="100", i_img=paths["i_img"], task_name=TAG,
                                              verbose=False)

    assert centerline_f == paths["manual_centerline"]
    assert mask_f == paths["mask"]


def test_manual_mode_writes_where_the_lookup_reads(sandbox):
    """`manual=True` must put the centerline where the manual-centerline lookup finds it.

    Regression for the `"{ses_name}"` literal, which sent the viewer's output to a
    directory named `{ses_name}` that nothing ever reads.
    """
    instance, paths, calls = sandbox

    instance.moco_mask(ID="100", i_img=paths["i_img"], task_name=TAG, manual=True, verbose=False)

    centerline_cmds = [c for c in calls if c.startswith("sct_get_centerline")]
    assert centerline_cmds, "sct_get_centerline was never called"
    written = centerline_cmds[0].split(" -o ")[1].split(" ")[0] + ".nii.gz"
    assert written == paths["manual_centerline"]
    assert "{ses_name}" not in written
