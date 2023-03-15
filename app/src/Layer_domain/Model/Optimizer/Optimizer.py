from abc import ABC
from abc import abstractmethod

class Optimizer(ABC):

    @property
    @staticmethod
    @abstractmethod
    def get_optimizer(learning_rate : float):
        pass