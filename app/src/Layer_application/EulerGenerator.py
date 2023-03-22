from abc import ABC
from abc import abstractmethod

from ..Layer_domain.Decorators.Timer import Timer

class EulerGenerator(ABC):
    """
    A class representing the EulerGenerator abstract base class for generating samples.

    Attributes
    ----------
    _Folder_path : str
        A string representing the folder path.
    _Number_of_objects : int
        An integer representing the number of objects.
    _Height : int
        An integer representing the height of the sample.
    _Width : int
        An integer representing the width of the sample.
    _Model : str
        A string representing the model used.

    Methods
    -------
    generate_euler_samples_random()
        Abstract method to generate Euler samples randomly.
    generate_euler_samples_settings()
        Abstract method to generate Euler samples with specific settings.
    """

    def __init__(self, 
                 Folder_path: str, 
                 Number_of_objects : int,
                 Height : int,
                 Width : int,
                 Model : str 
                 ) -> None:
        """
        Constructor for the EulerGenerator class.

        Parameters
        ----------
        Folder_path : str
            A string representing the folder path.
        Number_of_objects : int
            An integer representing the number of objects.
        Height : int
            An integer representing the height of the sample.
        Width : int
            An integer representing the width of the sample.
        Model : str
            A string representing the model used.
        """

        self._Folder_path = Folder_path;

        self._Number_of_objects = Number_of_objects;
        self._Number_of_objects = int(self._Number_of_objects);

        self._Height = Height;
        self._Height = int(self._Height);

        self._Width = Width;
        self._Width = int(self._Width);

        self._Model = Model;

    @Timer.timer
    @abstractmethod
    def generate_euler_samples_random():
        """
        Abstract method to generate Euler samples randomly.
        """
        pass
    
    @Timer.timer
    @abstractmethod
    def generate_euler_samples_settings():
        """
        Abstract method to generate Euler samples with specific settings.
        """
        pass