import torch
from torch import nn
from torch.nn import functional as F

from .base import ModelBase


class QuadraFC(ModelBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = QuardaFCNET(
            layer_num=5,
            units=20,
            **kwargs,
        )
        self.opt = torch.optim.Adam(self.model.parameters(), lr=2e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt,
            mode='max',
            patience=4,
            factor=0.4,
            verbose=True,
        )


class QuardaFCNET(nn.Module):

    def __init__(self, num_class, layer_num, units, use_batchnorm=False, **kwargs):
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
        self.out_layer = QuadraLayer(self.units[-1], num_class)

    def forward(self, x):
        for layer, bn in zip(self.layers, self.batchnorms):
            x = layer(x)
            if self.use_batchnorm:
                x = bn(x)
            # x = F.relu(x)
            x = F.leaky_relu(x, 0.2)

        x = self.out_layer(x)
        return x


class QuadraLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mean = nn.Parameter(torch.randn(out_channels, in_channels))
        self.log_var = nn.Parameter(torch.zeros(out_channels, 1))
        self.w = nn.Parameter(torch.randn(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        assert x.dim() == 2, x.dim()
        assert x.shape[1] == self.in_channels

        out = torch.zeros(len(x), self.out_channels)
        for i in range(self.out_channels):
            out[:, i] = 1. / torch.sqrt(torch.sum((x - self.mean[i]) ** 2, dim=-1))

        assert out.dim() == 2
        assert out.shape[1] == self.out_channels
        # out += self.b
        # out = out / torch.mean(out, dim=-1).view(len(out), 1)
        return out
