from abc import ABC
from abc import abstractmethod

from keras.models import Sequential

class ModelBuilder(ABC):
    """
    An abstract class that defines an interface for building Keras models.

    Methods:
    --------
    build_model() -> Sequential:
        Builds and returns a Keras Sequential model.

    """

    @abstractmethod
    def build_model(self) -> Sequential:
        """
        Abstract method that must be implemented by subclasses to build a Keras Sequential model.

        Returns:
        --------
        A Keras Sequential model.
        """
        pass
