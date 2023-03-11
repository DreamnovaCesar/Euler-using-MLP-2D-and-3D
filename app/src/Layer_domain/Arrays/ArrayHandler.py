from abc import ABC
from abc import abstractmethod

import numpy as np

class ArrayHandler(ABC):
    """
    An abstract base class for handling arrays.

    """
    @abstractmethod
    def get_number(self, Storage_list: list[str]) -> np.ndarray:
        """
        Convert a list of strings to a numpy array of numbers.

        Parameters
        ----------
        Storage_list : list[str]
            A list of strings to convert to numbers.

        Returns
        -------
        np.ndarray
            A numpy array of numbers.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        """
        pass