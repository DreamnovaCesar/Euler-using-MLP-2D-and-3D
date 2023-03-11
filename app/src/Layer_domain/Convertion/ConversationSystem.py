from abc import ABC
from abc import abstractmethod

class ConvertionSystem(ABC):
    """
    An abstract base class for conversion systems.
    """
    @staticmethod
    @abstractmethod
    def convertion_system(Value_to_convert: int) -> str:
        """
        Convert the given value to a string representation in a specific
        conversion system.

        Parameters
        ----------
        Value_to_convert : int
            The value to convert.

        Returns
        -------
        str
            A string representation of the converted value.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        """
        pass