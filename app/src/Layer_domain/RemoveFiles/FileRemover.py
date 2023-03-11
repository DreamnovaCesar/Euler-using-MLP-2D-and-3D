from abc import ABC
from abc import abstractmethod

class FileRemover(ABC):
    """
    An abstract base class for removing files in a given folder path.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the files to be removed.
    
    Methods
    ----------
    remove_files(): Removes all files and subdirectories in the specified folder.
    """

    def __init__(self, folder_path: str) -> None:
        """
        Constructor method for the FileRemover class.

        """
        self.folder_path = folder_path

    @abstractmethod
    def remove_files(self) -> None:
        """
        Abstract method to be implemented by subclasses to remove files from the folder path.
        """
        pass