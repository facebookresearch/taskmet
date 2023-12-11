# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn


class Task(object):
    def __init__(self, cfg):
        super().__init__()

    def decision(self, y):
        """
        Given model prediction output, which parameterize the downstream task,
        outputs the decision variable.
        z^* = argmin_z g(z,y)

        Args:
            y: model prediction output

        Returns:
            z*: decision variable
        """
        pass

    def objective(self, y, z):
        """
        Given model prediction output and decision variable, 
        outputs the objective value.
        obj = g(z, y)

        Args:
            y: model prediction output
            z: decision variable

        Returns:
            obj: objective value
        """
        pass

    def task_loss(self, pred, y, isTrain=True, **kwargs):
        """
        Value of objective function, under true y parameters
        and z find using predicted y parameters

        Args:
            pred: predicted y parameters
            y: true y parameters
            isTrain: whether in training mode

        Returns:
            loss: objective value
        """

        Zs = self.decision(pred, isTrain=isTrain, **kwargs)
        obj = self.objective(y, Zs, isTrain=isTrain, **kwargs)
        
        return -obj
