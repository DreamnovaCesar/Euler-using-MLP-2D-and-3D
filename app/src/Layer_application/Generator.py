from abc import ABC
from abc import abstractmethod

class Generator(ABC):
    """Abstract base class for generators.
    
    This class defines an abstract base class for generators. It is meant to be subclassed
    and extended to implement concrete generators. The class defines a single abstract
    method, `generator`, which should be implemented by concrete subclasses.
    """

    @property
    @staticmethod
    @abstractmethod
    def generator(self):
        """Abstract method to generate data.
        
        This method should be implemented by concrete subclasses to generate data.
        
        Returns
        -------
        data : ndarray
            The generated data.
        """
        pass