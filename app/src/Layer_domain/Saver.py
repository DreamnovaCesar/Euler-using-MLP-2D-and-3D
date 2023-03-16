import os

from abc import ABC
from abc import abstractmethod

class Saver(ABC):

    @abstractmethod
    def save_file(self, File_name):
        pass