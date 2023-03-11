from abc import ABC
from abc import abstractmethod

class JsonOperations(ABC):
    """
    An abstract class for performing JSON file operations.

    Attributes
    ----------
    None

    Methods
    -------
    create_json_file(self, data: dict, file_path: str) -> None:
        Abstract method to create a JSON file and write the given data to it.
        This method must be implemented by the child classes.

    read_json_file(self, file_path: str) -> dict:
        Abstract method to read a JSON file and return its data as a dictionary.
        This method must be implemented by the child classes.

    """
    @staticmethod
    @abstractmethod
    def create_json_file(data: dict, file_path: str) -> None:
        """
        Abstract method to create a JSON file and write the given data to it.

        Parameters
        ----------
        data : dict
            A dictionary containing the data to be written to the file.
        file_path : str
            The file path of the JSON file.

        Returns
        -------
        None
        """
        pass
    
    @staticmethod
    @abstractmethod
    def read_json_file(file_path: str) -> dict:
        """
        Abstract method to read a JSON file and return its data as a dictionary.

        Parameters
        ----------
        file_path : str
            The file path of the JSON file.

        Returns
        -------
        dict
            A dictionary containing the data from the JSON file.
        """
        pass