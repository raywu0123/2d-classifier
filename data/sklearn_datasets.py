from sklearn.datasets import make_moons, make_circles, make_classification, make_blobs

from .base import DataProviderBase


class SklearnMoons(DataProviderBase):

    def __init__(self, noise=0.05, random_state=0):
        self.noise = noise
        self.random_state = random_state

    def get_data(self, num_points=10000):
        data, label = make_moons(num_points, noise=self.noise, random_state=self.random_state)
        return data[:, 0], data[:, 1], label

    def data_format(self):
        return {
            'num_class': 2
        }


class SklearnCircles(DataProviderBase):

    def __init__(self, noise=0.05, factor=0.5, random_state=0):
        self.noise = noise
        self.factor = factor
        self.random_state = random_state

    def get_data(self, num_points=10000):
        data, label = make_circles(
            num_points,
            noise=self.noise, factor=self.factor, random_state=self.random_state
        )
        return data[:, 0], data[:, 1], label

    def data_format(self):
        return {
            'num_class': 2
        }


class SklearnBlobs(DataProviderBase):

    def __init__(self, num_class=2, random_state=0):
        self.num_class = num_class
        self.random_state = random_state

    def get_data(self, num_points=10000):
        data, label = make_blobs(
            num_points,
            n_features=2,
            centers=self.num_class,
            cluster_std=1. / self.num_class,
        )
        return data[:, 0], data[:, 1], label

    def data_format(self):
        return {
            'num_class': self.num_class
        }


class SklearnClassification(DataProviderBase):

    def __init__(self, num_class=2, random_state=0):
        self.num_class = num_class
        self.random_state = random_state

    def get_data(self, num_points=10000):
        data, label = make_classification(
            n_samples=num_points,
            n_features=2,
            n_classes=self.num_class,
            n_redundant=0,
            n_informative=2,
            n_clusters_per_class=1
        )
        return data[:, 0], data[:, 1], label

    def data_format(self):
        return {
            'num_class': self.num_class
        }
