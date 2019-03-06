from abc import ABC, abstractmethod


class DataProviderBase(ABC):

    @abstractmethod
    def get_data(self, num_points):
        pass

    @property
    @abstractmethod
    def data_format(self):
        pass
