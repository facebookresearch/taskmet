#!/usr/bin/env python3

import glob
import os
import json
import yaml
import pandas as pd
import pickle as pkl
import torch
from workspace import Workspace as W
import numpy as np
from utils import print_metrics, init_if_not_saved, move_to_gpu
from losses import MSE, get_loss_fn
from models import model_dict, MetricModel


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

    fname = exp_path + "/best.pkl"
    if not os.path.exists(fname):
        continue

    with open(fname, "rb") as f:
        workspace = pkl.load(f)

    # loss_fn = get_loss_fn(
    #     workspace.cfg.loss if not isinstance(workspace.model, MetricModel) else "dfl",
    #     workspace.problem,
    #     **dict(workspace.cfg.loss_kwargs),
    # )

    move_to_gpu(workspace.problem)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    workspace.model = workspace.model.to(device)

    # X_train, Y_train, Y_train_aux = workspace.problem.get_train_data()
    X_val, Y_val, Y_val_aux = workspace.problem.get_val_data()
    # X_test, Y_test, Y_test_aux = workspace.problem.get_test_data()

    # Decision Quality
    pred = workspace.model(X_val).squeeze()
    mse = (pred - Y_val).pow(2).mean().item()

    row.update({"mse": mse})

    # row["id"] = int(exp_path.split("/")[-1][0])
    if int(row["seed"]) == 3:
        continue
    rows.append(row)

print("Number of experiments loaded: ", len(rows))
if len(rows) == 0:
    import sys

    sys.exit(-1)

df = pd.DataFrame(rows)
df = df.apply(pd.to_numeric, errors="ignore")
df = df.sort_values(by=["dataset_seed", "seed"])
print(df)

# if "train_dq_norm" in df.columns:
#     print("== mean train DQ")
#     print(df.groupby(["problem", "method"])["train_dq_norm"].agg(["mean", "std"]))
#     print()

# if "val_dq_norm" in df.columns:
#     print("== mean val DQ")
#     print(df.groupby(["problem", "method"])["val_dq_norm"].agg(["mean", "std"]))
#     print()
# print(df.groupby('problem')['test_dq_norm'].agg(['mean', 'std']))
print("== mean test MSE")
print(df.groupby(["problem", "method"])["mse"].agg(["mean", "std"]))
