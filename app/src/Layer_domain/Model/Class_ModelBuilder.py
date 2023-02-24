from abc import ABC
from abc import abstractmethod

from keras.models import Sequential

class ModelBuilder(ABC):

    @abstractmethod
    def build_model(self) -> Sequential:
        pass
