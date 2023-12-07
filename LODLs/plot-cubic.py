#!/usr/bin/env python3

import argparse

import torch
import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt

from utils import move_to_gpu

from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_root", type=str)
    parser.add_argument("--pkl_tag", type=str, default="latest")
    args = parser.parse_args()

    exp_path = f"{args.exp_root}/{args.pkl_tag}.pkl"
    assert os.path.exists(exp_path)
    print("-- loading exp")
    with open(exp_path, "rb") as f:
        exp = pkl.load(f)
    print("-- done")

    if torch.cuda.is_available():
        move_to_gpu(exp.problem)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        exp.model = exp.model.to(device)

    X_train, Y_train, Y_train_aux = exp.problem.get_train_data()
    exp.model.update_predictor(X_train, Y_train, num_iters=1)

    loc = args.exp_root+"/latest.png"
    exp.problem.plot(loc, exp) #, show_grad=True)



if __name__ == "__main__":
    main()
