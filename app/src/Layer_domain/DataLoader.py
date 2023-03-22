import numpy as np

from abc import ABC
from abc import abstractmethod


class DataLoader(ABC):
    """
    A class that loads data from files.
    
    Methods
    -------
    load_data(file_path)

    """
    
    @property
    @staticmethod
    @abstractmethod
    def load_data(file_path):
        pass