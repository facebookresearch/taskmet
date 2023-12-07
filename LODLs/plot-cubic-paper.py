#!/usr/bin/env python3

import argparse

import torch
import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt

plt.style.use("bmh")
plt.rc("text", usetex=True)

from utils import move_to_gpu

from IPython.core import ultratb

sys.excepthook = ultratb.FormattedTB(mode="Verbose", color_scheme="Linux", call_pdb=1)


def load(path):
    assert os.path.exists(path)
    with open(path, "rb") as f:
        exp = pkl.load(f)
    if torch.cuda.is_available():
        move_to_gpu(exp.problem)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        exp.model = exp.model.to(device)
    return exp

def main():
    # exp_metric = load("exp/local/2023.05.15/1515/best.pkl")
    exp_metric = load("exp/local/2023.05.15/1542/best.pkl")
    exp_mse = load("exp/local/2023.05.15/1518/best.pkl")

    X_train, Y_train, Y_train_aux = exp_metric.problem.get_train_data()
    exp_metric.model.update_predictor(X_train, Y_train, num_iters=1)

    loc = 'cubic-metric.pdf'

    x = torch.linspace(-1.0, 1.0, steps=100).cuda().unsqueeze(1)
    y = 10 * (x.pow(3) - 0.65 * x).squeeze()

    nrow, ncol = 2, 1
    fig, axs = plt.subplots(
        nrow, ncol, figsize=(4 * ncol, 2 * nrow), height_ratios=[2.5, 1]
    )
    ax = axs[0]
    ax.plot(x.ravel().cpu(), y.ravel().cpu(), color="k", label="target")

    y_pred = exp_metric.model(x.to("cuda")).cpu().detach().numpy()
    ax.plot(x.ravel().cpu(), y_pred.ravel(), label="Metric")

    y_pred = exp_mse.model(x.to("cuda")).cpu().detach().numpy()
    l, = ax.plot(x.ravel().cpu(), y_pred.ravel(), label="MSE")
    mse_color = l.get_color()

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y(x)$")
    ax.set_title("model predictions")

    ax = axs[1]
    metric_preds = exp_metric.model.metric_forward(x).cpu().detach().numpy().ravel()
    ax.plot(x.ravel().cpu(), metric_preds, label="metric")
    ax.axhline(1., label='MSE', color=mse_color)
    ax.set_xlabel("$x$")

    ax.set_ylabel("$A(x)$")
    # ax.set_ylim(0.0, max(1.0, 1.1 * metric_preds.max()))


    ax.set_title("metric values")

    fig.tight_layout()
    fig.savefig(loc, bbox_inches="tight", pad_inches=0.0, dpi=300, transparent=True)
    plt.close(fig)
    print(f'Plot saved to {loc}')

if __name__ == "__main__":
    main()
