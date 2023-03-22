from abc import ABC
from abc import abstractmethod

class Optimizer(ABC):
    """
    Abstract base class for optimizers.

    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.

    """

    @property
    @staticmethod
    @abstractmethod
    def get_optimizer(learning_rate : float):
        """
        Abstract method that creates and returns a new optimizer instance.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        """
        pass