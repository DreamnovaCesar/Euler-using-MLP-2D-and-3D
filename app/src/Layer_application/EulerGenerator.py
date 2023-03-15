from abc import ABC
from abc import abstractmethod

from ..Layer_domain.Decorators.Timer import Timer

class EulerGenerator(ABC):

    def __init__(self, 
                 Folder_path: str, 
                 Number_of_objects : int,
                 Height : int,
                 Width : int) -> None:

        self._Folder_path = Folder_path
        self._Number_of_objects = Number_of_objects
        self._Number_of_objects = int(self._Number_of_objects);
    
        self._Height = Height
        self._Width = Width

    @Timer.timer
    @abstractmethod
    def generate_euler_samples_random():
        pass
    
    @Timer.timer
    @abstractmethod
    def generate_euler_samples_settings():
        pass