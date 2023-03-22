from abc import ABC
from abc import abstractmethod

class DataFrameCreator(ABC):
    """
    An abstract base class that defines the interface for creating 
    a pandas DataFrame from historical data.
    """

    @staticmethod
    @abstractmethod
    def create_dataframe_history() -> None:
        """
        An abstract method that creates a pandas DataFrame from historical data.
        """
        pass