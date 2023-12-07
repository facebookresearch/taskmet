#!/usr/bin/env python

import os
import sys
from IPython.core import ultratb

# sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)

# Makes sure hashes are consistent
hashseed = os.getenv("PYTHONHASHSEED")
if not hashseed:
    os.environ["PYTHONHASHSEED"] = "0"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from workspace import Workspace as W

import hydra
import pickle as pkl


@hydra.main(config_path="config", config_name="main.yaml", version_base="1.1")
def main_function(cfg):
    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        print(f"Resuming fom {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
    else:
        workspace = W(cfg)

    workspace.run()

    print("Training complete, testing best model")
    del workspace

    fname = os.getcwd() + "/best.pkl"
    assert os.path.exists(fname)
    with open(fname, "rb") as f:
        workspace = pkl.load(f)

    workspace.test()


if __name__ == "__main__":
    main_function()
