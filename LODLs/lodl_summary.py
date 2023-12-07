#!/usr/bin/env python3

import glob
import os
import json
import yaml
import numpy as np
import pandas as pd

# pd.set_option("display.precision", 2)
import argparse

import sys
from IPython.core import ultratb

# sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument("exp_root", type=str)
args = parser.parse_args()

exp_paths = glob.glob(args.exp_root + "/*")

# nest through all the directoryies inside exp_paths until we find the results.json file
# then load the results.json file and append it to a list of results

rows = []
for exp_path in exp_paths:
    dataset_seed = exp_path.split("/")[-1][-1]
    dirs = glob.glob(exp_path + "/*")
    for dir in dirs:
        row = {}
        seed = dir.split("/")[-1]
        # if int(seed) > 4:
        #     continue
        filename = dir + "/results.json"
        with open(filename, "r") as f:
            stats = json.load(f)
            metrices = stats["best_loss_metrices"]
            for key in ["train", "val", "test"]:
                metrices[key]["objective"] = (
                    metrices[key]["objective"] - stats["random"]
                ) / (stats["optimal"] - stats["random"])
                row.update({key + "_" + k: v for k, v in metrices[key].items()})
        row["seed"] = seed
        row["dataset_seed"] = dataset_seed
        rows.append(row)

print("Number of experiments loaded: ", len(rows))
if len(rows) == 0:
    print("No experiments found")

df = pd.DataFrame(rows)
df = df.apply(pd.to_numeric, errors="ignore")
df = df.sort_values(by=["dataset_seed", "seed"])
print(df)


print("== mean train DQ")
print(df["train_objective"].agg(["mean", "std"]))
print()

print("== mean val DQ")
print(df["val_objective"].agg(["mean", "std"]))
print()

print("== mean test DQ")
print(df["test_objective"].agg(["mean", "std"]))

# .groupby(["dataset_seed"])
