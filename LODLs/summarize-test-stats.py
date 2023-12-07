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

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


parser = argparse.ArgumentParser()
parser.add_argument("exp_root", type=str)
args = parser.parse_args()

exp_paths = glob.glob(args.exp_root + "/*")

rows = []
for exp_path in exp_paths:
    row = {}

    fname = exp_path + "/.hydra/overrides.yaml"
    if not os.path.exists(fname):
        continue

    with open(fname, "r") as f:
        overrides = yaml.load(f, Loader=yaml.Loader)
        overrides = dict([x.split("=") for x in overrides])
        filter_keys = []
        # overrides = {
        #     clean_key(k): v for (k,v) in overrides.items()
        #     if k not in filter_keys}
        row.update(overrides)

    fname = exp_path + "/test_stats.json"
    if not os.path.exists(fname):
        continue

    with open(fname, "r") as f:
        stats = json.load(f)
        row.update(stats)

    # row["id"] = int(exp_path.split("/")[-1][0])
    # if int(row["seed"]) > 4:
    #     continue
    rows.append(row)

print("Number of experiments loaded: ", len(rows))
if len(rows) == 0:
    import sys

    sys.exit(-1)

df = pd.DataFrame(rows)
df = df.apply(pd.to_numeric, errors="ignore")
df = df.sort_values(by=["dataset_seed", "seed"])
print(df)

if "train_dq_norm" in df.columns:
    print("== mean train DQ")
    print(df.groupby(["problem", "method"])["train_dq_norm"].agg(["mean", "std"]))
    print()

if "val_dq_norm" in df.columns:
    print("== mean val DQ")
    print(df.groupby(["problem", "method"])["val_dq_norm"].agg(["mean", "std"]))
    print()
# print(df.groupby('problem')['test_dq_norm'].agg(['mean', 'std']))
print("== mean test DQ")
print(df.groupby(["problem", "method"])["test_dq_norm"].agg(["mean", "std"]))
