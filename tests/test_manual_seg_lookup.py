# -*- coding: utf-8 -*-
"""Tests for `find_manual_sc_seg()`.

Backs the manual-preferred MOTOR segmentation lookup in `epi_derive_seg_from_rest`: without
it, the warped REST segmentation overwrites the canonical path on every run and a manually
corrected MOTOR segmentation never takes effect (#98, same failure mode as #113).

Run with the pipeline's Python (see tests/test_moco_mask.py for why):

    $SCT_DIR/python/envs/venv_sct/bin/python3 -m pytest tests/ -v
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "code"))

from preprocess import find_manual_sc_seg  # noqa: E402

TAG_3MM = "task-motor_acq-shimSlice+3mm"
TAG_1MM = "task-motor_acq-shimSlice+1mm+sms2"


@pytest.fixture
def manual_dir(tmp_path):
    d = tmp_path / "derivatives" / "manual"
    d.mkdir(parents=True)
    return str(d)


def _touch(manual_dir, ID, name):
    p = os.path.join(manual_dir, f"sub-{ID}", "func", name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w").close()
    return p


def test_returns_none_when_absent(manual_dir):
    assert find_manual_sc_seg("100", TAG_3MM, manual_dir) is None


def test_finds_segmentation_without_run_entity(manual_dir):
    want = _touch(manual_dir, "100", f"sub-100_{TAG_3MM}_bold_moco_mean_label-SC_seg.nii.gz")
    assert find_manual_sc_seg("100", TAG_3MM, manual_dir) == want


def test_finds_segmentation_with_run_entity(manual_dir):
    """The MOTOR mean often carries run-01/run-02, so the lookup must tolerate it."""
    want = _touch(manual_dir, "103", f"sub-103_{TAG_3MM}_run-01_bold_moco_mean_label-SC_seg.nii.gz")
    assert find_manual_sc_seg("103", TAG_3MM, manual_dir) == want


def test_picks_first_run_when_several(manual_dir):
    """Matches copy_segmentation_from_ref_tag(), which also takes the sorted-first run."""
    first = _touch(manual_dir, "105", f"sub-105_{TAG_3MM}_run-01_bold_moco_mean_label-SC_seg.nii.gz")
    _touch(manual_dir, "105", f"sub-105_{TAG_3MM}_run-02_bold_moco_mean_label-SC_seg.nii.gz")
    assert find_manual_sc_seg("105", TAG_3MM, manual_dir) == first


def test_does_not_confuse_similar_acquisition_tags(manual_dir):
    """`shimSlice+3mm` must not match a `shimSlice+3mm+avg3mm`-style derived tag, and the 1mm
    tag must not be returned for a 3mm query."""
    _touch(manual_dir, "100", f"sub-100_{TAG_1MM}_bold_moco_mean_label-SC_seg.nii.gz")
    assert find_manual_sc_seg("100", TAG_3MM, manual_dir) is None
    assert find_manual_sc_seg("100", TAG_1MM, manual_dir) is not None


def test_does_not_match_other_subjects(manual_dir):
    _touch(manual_dir, "106", f"sub-106_{TAG_3MM}_bold_moco_mean_label-SC_seg.nii.gz")
    assert find_manual_sc_seg("100", TAG_3MM, manual_dir) is None


def test_ignores_csf_label(manual_dir):
    """Only the SC label is a cord segmentation; CSF segmentations live alongside it."""
    _touch(manual_dir, "101", f"sub-101_{TAG_3MM}_bold_moco_mean_label-CSF_seg.nii.gz")
    assert find_manual_sc_seg("101", TAG_3MM, manual_dir) is None


def test_ignores_centerline_files(manual_dir):
    _touch(manual_dir, "100", f"sub-100_{TAG_3MM}_bold_tmean_centerline.nii.gz")
    assert find_manual_sc_seg("100", TAG_3MM, manual_dir) is None
