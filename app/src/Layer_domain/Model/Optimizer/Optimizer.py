from abc import ABC
from abc import abstractmethod

class Optimizer(ABC):

    @abstractmethod
    def get_optimizer(self, learning_rate: float = 0.0000001):
        pass