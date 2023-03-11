import numpy as np

from abc import ABC
from abc import abstractmethod


class DataLoader(ABC):
    """
    A class that loads data from files.
    """
    
    @staticmethod
    @abstractmethod
    def load_data(file_path):
        pass