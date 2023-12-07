#!/usr/bin/env python3

import torch

import models
import CubicTopK
import utils

problem = CubicTopK.CubicTopK()
utils.move_to_gpu(problem)

model = models.MetricModel(1, 1, 1, output_activation='linear').cuda()

X_train, Y_train, Y_train_aux = problem.get_train_data()
X_val, Y_val, Y_val_aux = problem.get_val_data()
X_test, Y_test, Y_test_aux = problem.get_test_data()

model.update_predictor(X_train, Y_train, num_iters=1001)
model.make_predictor_differentiable(X_train, Y_train)
preds = model(X_val[0])
g = torch.autograd.grad(preds[0,0], model.metric_params[1])
print(g)

model.implicit_diff_mode = 'torchopt_exact'
model.make_predictor_differentiable(X_train, Y_train)
preds = model(X_val[0])
g = torch.autograd.grad(preds[0,0], model.metric_params[1])
print(g)

import ipdb; ipdb.set_trace()
