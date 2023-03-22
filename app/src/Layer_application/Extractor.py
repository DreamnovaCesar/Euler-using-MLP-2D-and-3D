from abc import ABC
from abc import abstractmethod

class Extractor(ABC):
    """
    An abstract base class for data extractors.

    Attributes
    ----------
    None

    Methods
    -------
    extractor()
        An abstract method to be implemented by subclasses for extracting information from data.
    """

    @property
    @abstractmethod
    def extractor(self):
        """
        An abstract method to be implemented by subclasses for extracting information from data.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        pass