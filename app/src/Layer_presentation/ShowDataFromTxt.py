import numpy as np

from abc import ABC
from abc import abstractmethod

class ShowDataFromTxt(ABC):
    """
    A class that loads data from files.
    """

    @staticmethod
    @abstractmethod
    def show_data_from_file(Array : np.ndarray):
        pass