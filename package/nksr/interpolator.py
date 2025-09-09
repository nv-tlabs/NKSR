# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import math
import torch
import torch.nn as nn
import numpy as np


class MLPWithGrad(nn.Module):
    def __init__(self, n_inputs, n_outputs, n_x=0, n_layers=1, n_units=100, nonlinear=nn.ReLU, init=False,
                 n_fourier: int = 0, sigma_fourier: float = 0.1):
        """ The MLP has the first and last layers as FC.
        :param n_inputs: input dim
        :param n_outputs: output dim
        :param n_layers: layer num = n_layers + 2
        :param n_units: the dimension of hidden layers
        :param nonlinear: nonlinear function
        """
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_layers = n_layers
        self.n_units = n_units
        self.nonlinear = nonlinear
        self.n_fourier = n_fourier
        self.sigma_fourier = sigma_fourier

        if self.n_fourier > 0:
            layers = [nn.Linear(self.n_fourier * 2, self.n_units)]
            self.tpB = nn.Parameter(
                torch.randn((self.n_inputs + n_x, self.n_fourier)) * (sigma_fourier * 2 * np.pi), requires_grad=False)
        else:
            layers = [nn.Linear(self.n_inputs + n_x, self.n_units)]
            self.tpB = None

        # create layers
        for i in range(n_layers):
            layers.append(nonlinear())
            layers.append(nn.Linear(self.n_units, self.n_units))
        layers.append(nonlinear())
        last_fc = nn.Linear(self.n_units, self.n_outputs)
        if init:
            nn.init.zeros_(last_fc.weight)
            nn.init.constant_(last_fc.bias, 1. / math.sqrt(self.n_outputs))
        layers.append(last_fc)
        self.layers = nn.Sequential(*layers)

    @classmethod
    def inv_nonlinear(cls, nonlinear):
        """
        This will return the inverse of the nonlinear function, which is with input as the activation rather than z
        Currently only support sigmoid and tanh.
        """
        if nonlinear == nn.Tanh:
            inv = lambda x: 1 - x * x
        elif nonlinear == nn.Sigmoid:
            inv = lambda x: x * (1 - x)
        elif nonlinear == nn.ReLU:
            inv = lambda x: torch.where(x == 0, torch.zeros_like(x), torch.ones_like(x))
        else:
            assert False, '{} inverse function is not emplemented'.format(nonlinear)
        return inv

    def forward(self, x, pred_grad: torch.Tensor = None):
        """
        --> [FC] [A FC] [A FC]
        :param x: (bs, n_inputs)
        :param pred_grad: (optionally) (bs, n_inputs, M)
        :return: y (bs, n_outputs), [J (bs, n_outputs, M) (optionally)]
        """
        bs = x.size(0)
        if self.tpB is not None:
            tpBx = torch.matmul(x, self.tpB)
            ffn = [torch.cos(tpBx), torch.sin(tpBx)]
            x = torch.cat(ffn, dim=1)
            if pred_grad is not None:
                pred_grad = self.tpB.transpose(0, 1).unsqueeze(0).expand(bs, -1, -1) @ pred_grad
                pred_grad = torch.cat([-ffn[1].unsqueeze(-1) * pred_grad, ffn[0].unsqueeze(-1) * pred_grad], dim=1)
        for layer_i, layer in enumerate(self.layers):
            x = layer(x)
            if pred_grad is not None:
                if layer_i % 2 == 1:
                    pred_grad = self.inv_nonlinear(self.nonlinear)(x).unsqueeze(-1) * pred_grad
                else:
                    pred_grad = layer.weight.unsqueeze(0).expand(bs, -1, -1) @ pred_grad
        return x, pred_grad


class MLPFeatureInterpolator(nn.Module):
    def __init__(self, theta_dim: int, n_hidden: int, hidden_dim: int, **mlp_kwargs):
        super().__init__()
        self.mlp = MLPWithGrad(
            n_inputs=theta_dim, n_outputs=theta_dim, n_layers=n_hidden, n_units=hidden_dim,
            nonlinear=nn.ReLU, **mlp_kwargs
        )

    def interpolate(self, queries: torch.Tensor, grid, depth_data: torch.Tensor, grad: bool = False):
        ret = grid.sample_trilinear(queries, depth_data, return_grad=grad)
        if grad:
            return self.mlp(ret[0], ret[1])
        else:
            return self.mlp(ret)[0]
