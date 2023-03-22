
from abc import ABC
from abc import abstractmethod

class Saver(ABC):
    """
    Abstract base class for file savers.
    """

    @abstractmethod
    def save_file(
        self
    ):
        
        """
        Save a file with the given name and data.
        """
        pass