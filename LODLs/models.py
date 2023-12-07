from numpy import square
import torch
import functorch
import copy
import torch.nn as nn
from math import sqrt
from functools import reduce
import operator
import pdb
import random
import functools
import torchopt

from utils import View


# TODO: Pretty it up
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


class MetricNN(nn.Module):
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


class MetricModel(nn.Module):
    def __init__(
        self,
        num_features,
        num_targets,
        num_layers,
        intermediate_size=10,
        activation="relu",
        output_activation="sigmoid",
        prediction_batchsize=1,
        implicit_diff_batchsize=100,
        predictor_lr=1e-3,
        predictor_grad_norm_threshold=1e-3,
        implicit_diff_mode="exact",
        implicit_diff_iters=5,
        predictor_grad_clip_norm=1e-0,
        metric_kwargs={},
    ):
        super(MetricModel, self).__init__()
        self.predictor = dense_nn(
            num_features,
            num_targets,
            num_layers,
            intermediate_size=intermediate_size,
            activation=activation,
            output_activation=output_activation,
        )
        self.prediction_batchsize = prediction_batchsize
        self.predictor_grad_norm_threshold = predictor_grad_norm_threshold
        self.implicit_diff_batchsize = implicit_diff_batchsize
        self.implicit_diff_mode = implicit_diff_mode
        self.implicit_diff_iters = implicit_diff_iters
        self.predictor_grad_clip_norm = predictor_grad_clip_norm

        # TODO: add aux_data, e.g., Q?
        self.metric_def = MetricNN(num_features, num_targets, **metric_kwargs).cuda()
        metric_func, self.metric_params = functorch.make_functional(self.metric_def)

        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=predictor_lr,
        )
        self.pred_forward = self.predictor.forward

    def make_predictor_differentiable(self, X_train, Y_train, verbose=False):
        # Assume we have the optimal predictor. Make the predictions
        # differentiable w.r.t. the metric with the IFT by applying
        # a Newton update to the predictor's parameters.
        metric_loss = self.get_metric_loss()

        num_param_pred = sum(p.numel() for p in self.predictor.parameters())
        pred_func, pred_params = functorch.make_functional(self.predictor)

        def pred_loss(pred_params, metric_params):
            losses = []
            num_samples = min(self.implicit_diff_batchsize, len(X_train))
            for i in random.sample(range(len(X_train)), num_samples):
                pred = pred_func(pred_params, X_train[i]).squeeze()
                losses.append(
                    metric_loss(
                        X_train[i], pred, Y_train[i], metric_params=metric_params
                    )
                )
            loss = torch.stack(losses).mean()
            return loss

        if verbose:
            g = functorch.grad(pred_loss)(pred_params, self.metric_params)
            g = torch.cat([e.flatten() for e in g])
            print(f"implicit diff gradient norm: {g.norm()}")

        if self.implicit_diff_mode == "exact":
            H = functorch.hessian(pred_loss)(pred_params, self.metric_params)
            H = torch.cat(
                [torch.cat([e.flatten() for e in Hpart]) for Hpart in H]
            ).reshape(num_param_pred, num_param_pred)
            g = functorch.grad(pred_loss)(pred_params, self.metric_params)
            g = torch.cat([e.flatten() for e in g])
            newton_update = g.unsqueeze(1).cholesky_solve(H).squeeze()

            start_idx = 0
            new_params = []
            for p in pred_params:
                n = p.numel()
                update_p = newton_update[start_idx : start_idx + n].view_as(p)
                new_params.append(p - update_p)
        elif self.implicit_diff_mode.startswith("torchopt"):
            # TODO: Move?
            solver_mode = self.implicit_diff_mode[9:]
            if solver_mode == "exact":
                solver = torchopt.linear_solve.solve_inv()
            elif solver_mode == "cg":
                solver = torchopt.linear_solve.solve_cg(
                    maxiter=self.implicit_diff_iters, atol=0.0
                )
            elif solver_mode == "normal_cg":
                solver = torchopt.linear_solve.solve_normal_cg(
                    maxiter=self.implicit_diff_iters, atol=0.0, ridge=1e-5
                )
            elif solver_mode == "neumann_series":
                solver = torchopt.linear_solve.solve_inv(ns=True)
            else:
                assert False

            @torchopt.diff.implicit.custom_root(
                functorch.grad(pred_loss, argnums=0),
                argnums=1,
                solve=solver,
            )
            def solve(pred_params, metric_params):
                return pred_params

            new_params = solve(pred_params, self.metric_params)
            if torch.tensor([torch.isnan(param).any() for param in new_params]).any():
                print("WARNING: NaN in new_params")
                print(new_params)
                new_params = pred_params
        else:
            assert False

        self.pred_forward = functools.partial(pred_func, new_params)

    def update_predictor(self, X_train, Y_train, num_iters=1001, **kwargs):
        # Fit the predictor to the data with the current metric loss
        metric_loss = self.get_metric_loss()

        # print('-- updating predictor')
        for train_iter in range(num_iters):
            losses = []
            num_samples = min(self.prediction_batchsize, len(X_train))
            for i in random.sample(range(len(X_train)), num_samples):
                pred = self.predictor(X_train[i]).squeeze()
                losses.append(metric_loss(X_train[i], pred, Y_train[i]))
            loss = torch.stack(losses).mean()
            self.predictor_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.predictor.parameters(), self.predictor_grad_clip_norm
            )
            self.predictor_optimizer.step()

            g = torch.cat([p.grad.flatten() for p in self.predictor.parameters()])
            if g.norm() < self.predictor_grad_norm_threshold:
                break

            if train_iter == 0 or train_iter % 10 == 0 and kwargs.get("verbose", False):
                print(
                    f"inner iter {train_iter} loss: {loss.item():.2e} grad norm: {g.norm():.2e}"
                )

        self.make_predictor_differentiable(X_train, Y_train, **kwargs)
        return {"inner_loss": loss.item()}

    def parameters(self):
        return self.metric_params

    def forward(self, X):
        return self.pred_forward(X)

    def metric_forward(self, X):
        metric_func, _ = functorch.make_functional(self.metric_def)
        return metric_func(self.metric_params, X)

    def get_metric_loss(self):
        metric_func, _ = functorch.make_functional(self.metric_def)

        def metric_loss(X, Yhats, Ys, metric_params=None):
            if metric_params == None:
                metric_params = self.metric_params
            # A = metric_func(metric_params, X).ravel()
            A = metric_func(metric_params, X)
            err = (Yhats - Ys).view(A.shape[0], A.shape[1], 1)
            # print(A.shape, err.shape)
            return (err.transpose(1, 2) @ A @ err).mean()
            return (A * (Yhats - Ys) ** 2).mean()

        return metric_loss

    def __getstate__(self):
        d = copy.copy(self.__dict__)
        del d["pred_forward"]
        return d

    def __setstate__(self, d):
        self.__dict__ = d
        self.pred_forward = self.predictor.forward


class DenseLoss(torch.nn.Module):
    """
    A Neural Network-based loss function
    """

    def __init__(self, Y, num_layers=4, hidden_dim=100, activation="relu"):
        super(DenseLoss, self).__init__()
        # Save true labels
        self.Y = Y.detach().view((-1))
        # Initialise model
        self.model = torch.nn.Parameter(
            dense_nn(
                Y.numel(),
                1,
                num_layers,
                intermediate_size=hidden_dim,
                output_activation=activation,
            )
        )

    def forward(self, Yhats):
        # Flatten inputs
        Yhats = Yhats.view((-1, self.Y.numel()))

        return self.model(Yhats)


class WeightedMSE(torch.nn.Module):
    """
    A weighted version of MSE
    """

    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSE, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.min_val = min_val

        # Initialise paramters
        self.weights = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * self.weights.clamp(min=self.min_val)).mean(
            dim=-1
        )

        return weighted_mse


class WeightedMSEPlusPlus(torch.nn.Module):
    """
    A weighted version of MSE
    """

    def __init__(self, Y, min_val=1e-3):
        super(WeightedMSEPlusPlus, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhats):
        # Flatten inputs
        Yhat = Yhats.view((-1, self.Y.shape[0]))

        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(
            min=self.min_val
        )
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(
            min=self.min_val
        )
        weights = pos_weights + neg_weights

        # Compute MSE
        squared_error = (Yhat - self.Y).square()
        weighted_mse = (squared_error * weights).mean(dim=-1)

        return weighted_mse


class WeightedCE(torch.nn.Module):
    """
    A weighted version of CE
    """

    def __init__(self, Y, min_val=1):
        super(WeightedCE, self).__init__()
        # Save true labels
        self.Y_raw = Y.detach()
        self.Y = self.Y_raw.view((-1))
        self.num_dims = self.Y.shape[0]
        self.min_val = min_val

        # Initialise paramters
        self.weights_pos = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))
        self.weights_neg = torch.nn.Parameter(self.min_val + torch.rand_like(self.Y))

    def forward(self, Yhat):
        # Flatten inputs
        if len(self.Y_raw.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[: -len(self.Y_raw.shape)], self.num_dims))

        # Get weights for positive and negative components separately
        pos_weights = (Yhat > self.Y.unsqueeze(0)).float() * self.weights_pos.clamp(
            min=self.min_val
        )
        neg_weights = (Yhat < self.Y.unsqueeze(0)).float() * self.weights_neg.clamp(
            min=self.min_val
        )
        weights = pos_weights + neg_weights

        # Compute MSE
        error = torch.nn.BCELoss(reduction="none")(Yhat, self.Y.expand(*Yhat.shape))
        weighted_ce = (error * weights).mean(dim=-1)

        return weighted_ce


class WeightedMSESum(torch.nn.Module):
    """
    A weighted version of MSE-Sum
    """

    def __init__(self, Y):
        super(WeightedMSESum, self).__init__()
        # Save true labels
        assert len(Y.shape) == 2  # make sure it's a multi-dimensional input
        self.Y = Y.detach()

        # Initialise paramters
        self.msesum_weights = torch.nn.Parameter(torch.rand(Y.shape[0]))

    def forward(self, Yhats):
        # Get weighted MSE-Sum
        sum_error = (self.Y - Yhats).mean(dim=-1)
        row_error = sum_error.square()
        weighted_mse_sum = (row_error * self.msesum_weights).mean(dim=-1)

        return weighted_mse_sum


class TwoVarQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(self, Y):
        super(TwoVarQuadratic, self).__init__()
        # Save true labels
        self.Y = torch.nn.Parameter(Y.detach().view((-1)))

        # Initialise paramters
        self.alpha = torch.nn.Parameter(torch.tensor(0.5))
        self.beta = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, Yhat):
        """ """
        # Flatten inputs
        Yhat = Yhat.view((Yhat.shape[0], -1))

        # Difference of squares
        # Gives diagonal elements
        diag = (self.Y - Yhat).square().mean()

        # Difference of sum of squares
        # Gives cross-terms
        cross = (self.Y - Yhat).mean().square()

        return self.alpha * diag + self.beta * cross


class QuadraticPlusPlus(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self, Y, quadalpha=1e-3, **kwargs  # true labels  # regularisation weight
    ):
        super(QuadraticPlusPlus, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))
        self.num_dims = self.Y.shape[0]

        # Create quadratic matrices
        bases = torch.rand((self.num_dims, self.num_dims, 4)) / (
            self.num_dims * self.num_dims
        )
        self.bases = torch.nn.Parameter(bases)

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[: -len(self.Y_raw.shape)], self.num_dims))

        # Measure distance between predicted and true distributions
        diff = (self.Y - Yhat).unsqueeze(-2)

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = self._get_basis(Yhat).clamp(-10, 10)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1).squeeze()

        return quad + self.alpha * mse

    def _get_basis(self, Yhats):
        # Figure out which entries to pick
        #   Are you above or below the true label
        direction = (Yhats > self.Y).type(torch.int64)
        #   Use this to figure out the corresponding index
        direction_col = direction.unsqueeze(-1)
        direction_row = direction.unsqueeze(-2)
        index = (direction_col + 2 * direction_row).unsqueeze(-1)

        # Pick corresponding entries
        bases = self.bases.expand(*Yhats.shape[:-1], *self.bases.shape)
        basis = bases.gather(-1, index).squeeze()
        return torch.tril(basis)


class LowRankQuadratic(torch.nn.Module):
    """
    Model that copies the structure of MSE-Sum
    """

    def __init__(
        self,
        Y,  # true labels
        rank=2,  # rank of the learned matrix
        quadalpha=0.1,  # regularisation weight
        **kwargs,
    ):
        super(LowRankQuadratic, self).__init__()
        self.alpha = quadalpha
        self.Y_raw = Y.detach()
        self.Y = torch.nn.Parameter(self.Y_raw.view((-1)))

        # Create a quadratic matrix
        # basis = torch.tril(
        #     torch.rand((self.Y.shape[0], rank)) / (self.Y.shape[0] * self.Y.shape[0])
        # )
        basis = torch.tril(
            torch.rand((self.Y.shape[0], self.Y.shape[0]))
            / (self.Y.shape[0] * self.Y.shape[0])
        )
        self.basis = torch.nn.Parameter(basis)

    def forward(self, Yhat):
        # Flatten inputs
        if len(Yhat.shape) >= 2:
            Yhat = Yhat.view((*Yhat.shape[: -len(self.Y_raw.shape)], self.Y.shape[0]))

        # Measure distance between predicted and true distributions
        diff = self.Y - Yhat

        # Do (Y - Yhat)^T Matrix (Y - Yhat)
        basis = torch.tril(self.basis).clamp(-100, 100)
        quad = (diff @ basis).square().sum(dim=-1).squeeze()

        # Add MSE as regularisation
        mse = diff.square().mean(dim=-1)

        return quad + self.alpha * mse


model_dict = {"dense": dense_nn}
