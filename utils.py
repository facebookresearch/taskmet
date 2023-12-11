# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn
from functools import reduce
import operator

def dense_nn(
    num_features,
    num_targets,
    num_layers,
    intermediate_size=10,
    activation="relu",
    output_activation="sigmoid",
):
    if num_layers > 1:
        if intermediate_size is None:
            intermediate_size = max(num_features, num_targets)
        if activation == "relu":
            activation_fn = torch.nn.ReLU
        elif activation == "sigmoid":
            activation_fn = torch.nn.Sigmoid
        else:
            raise Exception("Invalid activation function: " + str(activation))
        net_layers = [torch.nn.Linear(num_features, intermediate_size), activation_fn()]
        for _ in range(num_layers - 2):
            net_layers.append(torch.nn.Linear(intermediate_size, intermediate_size))
            net_layers.append(activation_fn())
        if not isinstance(num_targets, tuple):
            net_layers.append(torch.nn.Linear(intermediate_size, num_targets))
        else:
            net_layers.append(
                torch.nn.Linear(intermediate_size, reduce(operator.mul, num_targets, 1))
            )
            net_layers.append(View(num_targets))
    else:
        if not isinstance(num_targets, tuple):
            net_layers = [torch.nn.Linear(num_features, num_targets)]
        else:
            net_layers = [
                torch.nn.Linear(num_features, reduce(operator.mul, num_targets, 1)),
                View(num_targets),
            ]

    if output_activation == "relu":
        net_layers.append(torch.nn.ReLU())
    elif output_activation == "sigmoid":
        net_layers.append(torch.nn.Sigmoid())
    elif output_activation == "tanh":
        net_layers.append(torch.nn.Tanh())
    elif output_activation == "softmax":
        net_layers.append(torch.nn.Softmax(dim=-1))
    elif output_activation == "elu":
        net_layers.append(torch.nn.ELU())

    return torch.nn.Sequential(*net_layers)


class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f"View{self.shape}"

    def forward(self, input):
        """
        Reshapes the input according to the shape saved in the view data structure.
        """
        batch_size = input.shape[:-1]
        shape = (*batch_size, *self.shape)
        out = input.view(shape)
        return out
