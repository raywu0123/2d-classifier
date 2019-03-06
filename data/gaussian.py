import numpy as np

from .base import DataProviderBase


class AffineGaussians(DataProviderBase):

    def __init__(self, d: float = 1., var: float = .05, num_class: int = 2):
        self.d = d
        self.var = var
        self.num_class = num_class
        self.means = self._get_means()

    def get_data(self, num_points=10000):
        label = np.random.choice(range(self.num_class), num_points)
        x = np.random.normal(
            loc=self.means[label],
            scale=self.var,
            size=num_points
        )
        y = np.zeros_like(x)
        return x, y, label

    def _get_means(self):
        return np.linspace(
            - (self.d / 2) * (self.num_class // 2),
            + (self.d / 2) * (self.num_class // 2),
            self.num_class,
        )

    def data_format(self):
        return {
            'num_class': self.num_class,
        }
