import numpy as np

from .base import DataProviderBase


class Angular(DataProviderBase):
    def __init__(self, num_class: int = 2, noise_scale=0.1):
        self.num_class = num_class
        self.direction_vectors = self._get_direction_vectors()
        self.noise_scale = noise_scale

    def get_data(self, num_points=10000):
        label = np.random.choice(range(self.num_class), num_points)
        magnitude = np.random.chisquare(3, size=(num_points, 1))
        data_points = magnitude * self.direction_vectors[label]
        noise = np.random.normal(scale=self.noise_scale, size=(num_points, 1))
        data_points += noise
        x = data_points[:, 0]
        y = data_points[:, 1]
        return x, y, label

    def _get_direction_vectors(self):
        vecs = []
        for i in range(self.num_class):
            theta = 2 * np.pi / self.num_class * i
            vecs.append([np.cos(theta), np.sin(theta)])

        return np.asarray(vecs)

    def data_format(self):
        return {
            'num_class': self.num_class,
        }
