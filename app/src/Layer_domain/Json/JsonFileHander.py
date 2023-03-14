import json
from .JsonOperations import JsonOperations
from typing import Dict

class JsonFileHandler(JsonOperations):
    """
    A class for handling JSON files.

    Attributes
    ----------
    None

    Methods
    -------
    create_json_file(self, data: dict, file_path: str) -> None:
        Create a new JSON file and write the given data to it.

    read_json_file(self, file_path: str) -> dict:
        Read a JSON file and return its data as a dictionary.

    """

    def create_json_file(data: dict, file_path: str) -> None:
        """
        Create a new JSON file and write the given data to it.

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
        with open(file_path, 'w') as file:
            json.dump(data, file);

    def read_json_file(file_path: str) -> Dict[str, object]:
        """
        Read a JSON file and return its data as a dictionary.

        Parameters
        ----------
        file_path : str
            The file path of the JSON file.

        Returns
        -------
        dict
            A dictionary containing the data from the JSON file.
        """
        with open(file_path, 'r') as file:
            data = json.load(file);
        return data