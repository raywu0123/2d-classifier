import torch
from torch import nn
from torch.nn import functional as F

from .base import ModelBase


class EB(ModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = EB_NET(
            num_class=self.num_class,
            layer_num=3,
            units=20,
        )
        self.opt = torch.optim.Adam(
            self.model.parameters(),
            lr=1e-3,
            weight_decay=1e-2,
        )


class EB_NET(nn.Module):

    def __init__(self, num_class, layer_num, units):
        super().__init__()
        self.layer_num = layer_num
        self.units = [2] + [units] * layer_num

        self.layers = nn.ModuleList(
            [nn.Linear(in_units, out_units)
             for in_units, out_units in zip(self.units[:-1], self.units[1:])]
        )
        self.batchnorms = nn.ModuleList(
            [nn.BatchNorm1d(n_units) for n_units in self.units[1:]]
        )
        self.out_layer = nn.Linear(self.units[-1], num_class)
        self.out_bn = nn.BatchNorm1d(num_class)
        self.A = nn.Parameter(torch.randn(2, requires_grad=True) + 5.)
        self.B = nn.Parameter(torch.randn(2, requires_grad=True))

    def forward(self, x):
        for layer, bn in zip(self.layers, self.batchnorms):
            x = layer(x)
            x = bn(x)
            x = F.relu(x)

        x = self.out_layer(x)
        return x
