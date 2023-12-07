from functools import partial
import os
import sys

import argparse
import ast
import torch
import random
import numpy as np
import json
import pdb
import matplotlib.pyplot as plt
from copy import deepcopy, copy

import hydra
import setproctitle
import pickle as pkl

from BudgetAllocation import BudgetAllocation
from BipartiteMatching import BipartiteMatching
from PortfolioOpt import PortfolioOpt
from RMAB import RMAB
from CubicTopK import CubicTopK
from models import model_dict, MetricModel
from losses import MSE, get_loss_fn
from utils import print_metrics, init_if_not_saved, move_to_gpu
from logger import Logger


class Workspace:
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg

        self.load_problem()

        # set these after loading the problem for reproducibility
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        ipdim, opdim = self.problem.get_modelio_shape()
        if self.cfg.loss == "metric":
            model_builder = MetricModel
            lr = cfg.metric_lr
        else:
            model_builder = model_dict[self.cfg.pred_model]
            lr = cfg.pred_lr
        self.model = model_builder(
            num_features=ipdim,
            num_targets=opdim,
            num_layers=self.cfg.layers,
            intermediate_size=500,
            output_activation=self.problem.get_output_activation(),
            **dict(self.cfg.model_kwargs),
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=cfg.loss_kwargs.weight_decay
        )
        self.train_iter = 0
        self.best_loss = float("inf")
        self.best_DQ = -float("inf")

    def run(self):
        logger = Logger(os.getcwd(), "log.csv")
        loss_fn = get_loss_fn(
            self.cfg.loss if not isinstance(self.model, MetricModel) else "dfl",
            self.problem,
            **dict(self.cfg.loss_kwargs),
        )

        #   Move everything to GPU, if available
        if torch.cuda.is_available():
            move_to_gpu(self.problem)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)

        # if hasattr(self.problem, "plot"):
        #     self.problem.plot("latest.png", self)

        # Get data
        X_train, Y_train, Y_train_aux = self.problem.get_train_data()
        X_val, Y_val, Y_val_aux = self.problem.get_val_data()
        # X_test, Y_test, Y_test_aux = self.problem.get_test_data()

        while self.train_iter <= self.cfg.iters:
            metrics = {}

            #  inner loop, update predictor
            if self.cfg.loss == "metric":
                num_iters = (
                    self.cfg.num_inner_iters
                    if self.train_iter > 0
                    else self.cfg.num_inner_iters_init
                )
                predictor_metric = self.model.update_predictor(
                    X_train, Y_train, num_iters=num_iters, verbose=self.cfg.verbose
                )
                metrics.update(predictor_metric)

            losses = []
            DQ = []
            mse = []
            for i in random.sample(
                range(len(X_train)), min(self.cfg.batchsize, len(X_train))
            ):
                pred = self.model(X_train[i]).squeeze()
                losses.append(
                    loss_fn(
                        pred,
                        Y_train[i],
                        aux_data=Y_train_aux[i],
                        partition="train",
                        index=i,
                    )
                )
                Zs_pred = self.problem.get_decision(
                    pred, aux_data=Y_train_aux[i], isTrain=True
                )
                DQ.append(
                    self.problem.get_objective(
                        Y_train[i], Zs_pred, aux_data=Y_train_aux[i]
                    ).item()
                )
                mse.append((pred - Y_train[i]).pow(2).mean().item())

            loss = torch.stack(losses).mean()
            self.optimizer.zero_grad()
            loss.backward()

            metrics.update(
                {
                    "outer_loss": loss.item(),
                    "DQ": np.mean(DQ),
                    "MSE": np.mean(mse),
                }
            )

            if self.cfg.verbose:
                print(f"Outer Iter: {self.train_iter}, DQ: {metrics['DQ']:.2e}")

            logger.log(metrics, iter=self.train_iter, partition="Train")

            if self.train_iter % self.cfg.valfreq == 0:
                metrics = self.val_metrices(loss_fn, X_val, Y_val, Y_val_aux)
                logger.log(metrics, iter=self.train_iter, partition="Val")
                # Save model if it's the best one
                if metrics["outer_loss"] < self.best_loss:
                    self.save("best")
                    self.best_loss = metrics["outer_loss"]
                    self.best_iter = self.train_iter
                    self.time_since_best = 0
                if metrics["DQ"] > self.best_DQ:
                    self.save("best_DQ")
                    self.best_DQ = metrics["DQ"]

            self.optimizer.step()  # putting this after the validation step to avoid using new metric for validation inner loss
            self.train_iter += 1

            if self.time_since_best > self.cfg.patience and self.cfg.earlystopping:
                print("Stopping early")
                break

            self.time_since_best += 1

        print("Training complete, best model saved at iter {}".format(self.best_iter))
        logger.close()
        self.save()

    def val_metrices(self, loss_fn, X_val, Y_val, Y_val_aux):
        losses = []
        DQ = []
        mse = []
        if self.cfg.loss == "metric":
            inner_loss = []
            metric_loss = self.model.get_metric_loss()
        for i in range(len(X_val)):
            pred = self.model(X_val[i]).squeeze()
            losses.append(
                loss_fn(
                    pred,
                    Y_val[i],
                    aux_data=Y_val_aux[i],
                    partition="validation",
                    index=i,
                ).item()
            )
            Zs_pred = self.problem.get_decision(
                pred, aux_data=Y_val_aux[i], isTrain=True
            )
            DQ.append(
                self.problem.get_objective(
                    Y_val[i], Zs_pred, aux_data=Y_val_aux[i]
                ).item()
            )
            mse.append((pred - Y_val[i]).pow(2).mean().item())
            if self.cfg.loss == "metric":
                inner_loss.append(metric_loss(X_val[i], pred, Y_val[i]).item())
        if self.cfg.loss == "metric":
            metrics = {
                "inner_loss": np.mean(inner_loss),
                "outer_loss": np.mean(losses),
                "DQ": np.mean(DQ),
                "MSE": np.mean(mse),
            }
        else:
            metrics = {
                "outer_loss": np.mean(losses),
                "DQ": np.mean(DQ),
                "MSE": np.mean(mse),
            }

        return metrics

    def test(self):
        # Document how well this trained model does
        print("\nBenchmarking Model...")

        loss_fn = get_loss_fn(
            self.cfg.loss if not isinstance(self.model, MetricModel) else "dfl",
            self.problem,
            **dict(self.cfg.loss_kwargs),
        )

        #   Move everything to GPU, if available
        if torch.cuda.is_available():
            move_to_gpu(self.problem)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model = self.model.to(device)

        X_train, Y_train, Y_train_aux = self.problem.get_train_data()
        X_val, Y_val, Y_val_aux = self.problem.get_val_data()
        X_test, Y_test, Y_test_aux = self.problem.get_test_data()

        # Print final metrics
        datasets = [
            (X_train, Y_train, Y_train_aux, "train"),
            (X_val, Y_val, Y_val_aux, "val"),
            (X_test, Y_test, Y_test_aux, "test"),
        ]
        metrics = print_metrics(
            datasets, self.model, self.problem, self.cfg.loss, loss_fn, "Final"
        )

        #   Document the value of a random guess
        random_dq = self.DQ(Y_test, Y_test_aux, model="random")
        print(f"\nRandom Decision Quality: {random_dq:.2f} (normalized: 0)")

        #   Document the optimal value
        optimal_dq = self.DQ(Y_test, Y_test_aux, model="optimal")
        print(f"Optimal Decision Quality: {optimal_dq:.2f} (normalized: 1)")
        print()
        # self.save()

        dq_range = optimal_dq - random_dq
        test_dq = metrics["test"]["objective"]
        train_dq = metrics["train"]["objective"]
        val_dq = metrics["val"]["objective"]

        normalized_test_dq = (test_dq - random_dq) / dq_range
        normalized_train_dq = (train_dq - random_dq) / dq_range
        normalized_val_dq = (val_dq - random_dq) / dq_range

        print(f"Normalized Train Decision Quality: {normalized_train_dq:.2f}")
        print(f"Normalized Val Decision Quality: {normalized_val_dq:.2f}")
        print(f"Normalized Test Decision Quality: {normalized_test_dq:.2f}")

        test_stats = {
            "random_dq_unnorm": random_dq,
            "optimal_dq_unnorm": optimal_dq,
            "train_dq_unnorm": train_dq,
            "val_dq_unnorm": val_dq,
            "test_dq_unnorm": test_dq,
            "train_dq_norm": normalized_train_dq,
            "val_dq_norm": normalized_val_dq,
            "test_dq_norm": normalized_test_dq,
            "best_iter": self.best_iter,
        }

        fname = "test_stats.json"
        print(f"writing to {fname}")
        with open(fname, "w") as f:
            json.dump(test_stats, f)

    def DQ(self, Y_test, Y_test_aux, model="random"):
        objs = []
        for _ in range(10):
            if model == "random":
                pred = torch.rand_like(Y_test)
            elif model == "optimal":
                pred = Y_test
            Z_test_rand = self.problem.get_decision(
                pred, aux_data=Y_test_aux, isTrain=False
            )
            objectives = self.problem.get_objective(
                Y_test, Z_test_rand, aux_data=Y_test_aux
            )
            objs.append(objectives)
        dq = torch.stack(objs).mean().item()
        return dq

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        with open(path, "wb") as f:
            pkl.dump(self, f)

    def __getstate__(self):
        d = copy(self.__dict__)
        del d["problem"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.load_problem()

    def load_problem(self):
        init_problem = partial(init_if_not_saved, load_new=self.cfg.loadnew)
        problem_cls = hydra.utils._locate(self.cfg.problem_cls)
        self.problem = init_problem(problem_cls, dict(self.cfg.problem_kwargs))
