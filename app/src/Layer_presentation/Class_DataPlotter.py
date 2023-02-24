from abc import ABC
from abc import abstractmethod

class DataPlotter(ABC):
    
    @abstractmethod
    def plot_data_loss(self, Hist_data: object) -> None:
        pass

    @abstractmethod
    def plot_data_accuracy(self, Hist_data: object) -> None:
        pass