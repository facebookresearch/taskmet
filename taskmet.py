# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Union, Optional, Callable
from utils import dense_nn, View
import functorch
import torchopt
import random
from metric import Metric

class Predictor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.model = dense_nn()
   
    def forward(self, x):
        return self.model(x)
    
class TaskMet(object):
    def __init__(self, cfg, predictor, task) -> None:
        self.metric, self.metric_param = functorch.make_functional(Metric())
        self.predictor, self.pred_param = functorch.make_functional(predictor)
        self.task = task
        self.predictor_optimizer = torchopt.FuncOptimizer(
            torchopt.chain([torchopt.adam(lr=1e-3), 
                            torchopt.clip(self.predictor_grad_clip_norm)]))
        self.metric_optimizer = torch.optim.Adam(self.metric.parameters(), lr=1e-3)
        
    def pred_loss(self, pred_param, metric_param, x, y):
        if metric_param is None:
            metric_param = self.metric_param
        A = self.metric(metric_param, x)
        yhat = self.predictor(pred_param, x)
        err = (yhat - y).view(A.shape[0], A.shape[1], 1)
        # print(A.shape, err.shape)
        return (err.transpose(1, 2) @ A @ err).mean()
    
    def pred_optimality(self, pred_param, metric_param, x, y):
        losses = []
        num_samples = min(self.implicit_diff_batchsize, len(x))
        for i in random.sample(range(len(x)), num_samples):
            pred = self.predictor(pred_param, x[i]).squeeze()
            losses.append(
                self.pred_loss(
                    pred_param, x[i], y[i], metric_params=metric_param
                )
            )
        loss = torch.stack(losses).mean()
        return loss
    
    def train_predictor(self, pred_param, metric_param, x, y, batch_size, num_iters, **kwargs):
        # Fit the predictor to the data with the current metric value
        for train_iter in range(num_iters):
            losses = []
            num_samples = min(batch_size, len(x))
            for i in random.sample(range(len(x)), num_samples):
                losses.append(self.pred_loss(pred_param, metric_param, x[i], y[i]))
            loss = torch.stack(losses).mean()
            pred_param = self.predictor_optimizer.step(loss, pred_param)
                  
            g = torch.cat([p.flatten() for p in torch.autograd.grad(loss, pred_param)])
            if g.norm() < self.predictor_grad_norm_threshold:
                break

            if train_iter == 0 or train_iter % 10 == 0 and kwargs.get("verbose", False):
                print(
                    f"inner iter {train_iter} loss: {loss.item():.2e} grad norm: {g.norm():.2e}"
                )
       
        @torchopt.diff.implicit.custom_root(
            functorch.grad(self.pred_optimality, argnums=1),
            argnums=1,
            solve=torchopt.linear_solve.solve_normal_cg(
                    maxiter=self.implicit_diff_iters, atol=0.0, ridge=1e-5),
            )
        def solve(pred_param, metric_param):
            return pred_param

        pred_param = solve(pred_param, metric_param)
        
        return pred_param, loss.item()
    
    def train(self, x, y, batch_size, iters):
        for iter in range(iters):
            self.pred_param = self.train_predictor(self.pred_param, self.metric_param, x, y)

            losses=[]
            for i in random.sample(
                range(len(x)), min(self.cfg.batchsize, len(x))
            ):
                pred = self.predictor(self.pred_param, x).squeeze()
                loss = self.task.task_loss(pred, y, isTrain=True, **kwargs)
                losses.append(loss)

            loss = torch.stack(losses).mean()
            self.metric_optimizer.zero_grad()
            loss.backward()

            self.metric_optimizer.step()

if __name__ == '__main__':
    # write some demo toy experiment for TaskMet
    pass
