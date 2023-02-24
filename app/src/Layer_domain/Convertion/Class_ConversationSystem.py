from abc import ABC
from abc import abstractmethod

class ConvertionSystem(ABC):

    @staticmethod
    @abstractmethod
    def convertion_system(Value_to_convert: int) -> str:
        pass