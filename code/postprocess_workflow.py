#!/usr/bin/env python
# coding: utf-8

# Description: Postprocessing workflow — computes tSNR maps from REST data.
# Must be run after preprocessing and before first-level GLM analysis.

import json, sys, os, argparse
import pandas as pd

path_code = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(path_code, 'config', 'config_spine_7t_fmri.json')) as config_file:
    config = json.load(config_file)

parser = argparse.ArgumentParser()
parser.add_argument("--ids", nargs='+', default=[""])
parser.add_argument("--verbose", default="False")
parser.add_argument("--redo", default="true")
parser.add_argument("--path-data", required=True)
args = parser.parse_args()

IDs = args.ids
redo = args.redo.lower() == "true"
path_data = os.path.abspath(args.path_data)

config["raw_dir"] = path_data
config["code_dir"] = path_code

participants_tsv = pd.read_csv(os.path.join(path_code, 'config', 'participants.tsv'), sep='\t', dtype={'participant_id': str})

if IDs == [""]:
    IDs = list(participants_tsv["participant_id"])

sys.path.append(os.path.join(path_code, "code"))
import postprocess

# tSNR is a measure of scanner/physiological noise — use REST data only so
# genuine BOLD activation during the motor task does not inflate the estimate.
config["design_exp"]["task_names"] = ["rest"]

print("=== Postprocessing: tSNR (rest) Start ===", flush=True)
print("Participant(s) included:", IDs, flush=True)
print("==========================================", flush=True)

tsnr_ana = postprocess.TSNR_main(config, IDs, redo)
tsnr_ana.generate_tsnr_maps_and_csv()
tsnr_ana.generate_tsnr_maps_derived()

print("=== Postprocessing: tSNR Done ===", flush=True)
