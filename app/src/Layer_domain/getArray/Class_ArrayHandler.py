from abc import ABC
from abc import abstractmethod

import numpy as np

class ArrayHandler(ABC):
    @abstractmethod
    def get_number(self, Qs: list[str]) -> np.ndarray:
        pass