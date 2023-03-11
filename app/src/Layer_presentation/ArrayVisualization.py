from abc import ABC
from abc import abstractmethod

class ArrayVisualization(ABC):
    """
    An abstract class that defines methods to visualize arrays.
    """
    
    @abstractmethod
    def show_array(self, data):
        pass