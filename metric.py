# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn

class Metric(nn.Module):
    def __init__(
        self,
        num_features,
        num_output,
        num_hidden,
        identity_init,
        identity_init_scale,
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, num_output * num_output),
        )
        self.identity_fac_log = torch.nn.parameter.Parameter(torch.zeros([]))
        if identity_init:
            last_layer = self.base[-1]
            last_layer.weight.data.div_(identity_init_scale)
            last_layer.bias.data = torch.eye(num_output).view(-1)

        self.num_output = num_output

    def forward(self, x):
        # A = torch.nn.functional.softplus(self.base(x))
        identity_fac = torch.exp(self.identity_fac_log)
        L = self.base(x)
        L = L.view(L.shape[0], self.num_output, self.num_output)
        A = (
            torch.bmm(L, L.transpose(1, 2))
            + identity_fac * torch.eye(self.num_output).repeat(x.shape[0], 1, 1).cuda()
        )
        # TODO: extend for PSD matrices with bounds from the
        # identity metric
        return A
