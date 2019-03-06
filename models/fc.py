import torch
from torch import nn
from torch.nn import functional as F

from .base import ModelBase


class FC(ModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = FC_NET(
            layer_num=5,
            units=20,
            **kwargs,
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-4)


class FC_NET(nn.Module):
    def __init__(self, num_class, layer_num, units, use_batchnorm=True, **kwargs):
        super().__init__()
        self.use_batchnorm = use_batchnorm
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

    def forward(self, x):
        for layer, bn in zip(self.layers, self.batchnorms):
            x = layer(x)
            if self.use_batchnorm:
                x = bn(x)
            x = F.relu(x)

        x = self.out_layer(x)
        return x
