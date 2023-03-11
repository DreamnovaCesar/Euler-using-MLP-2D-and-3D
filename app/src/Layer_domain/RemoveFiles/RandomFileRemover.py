import os
from random import sample

from .FileRemover import FileRemover

class RandomFileRemover(FileRemover):
    """
    A class that inherits from the `FileRemover` class and implements
    the `remove_files` method to remove a random selection of files from
    a specified folder path.

    Attributes:
    -----------
    folder_path: str
        The path of the folder to remove files from.

    Methods:
    --------
    remove_files(number_files_to_remove: int) -> None:
        Removes a random selection of files from the specified folder path,
        with the number of files to remove specified as an input argument.
        Prints the name of each removed file to the console.
    """

    def remove_files(self, number_files_to_remove: int) -> None:
        """
        Removes a random selection of files from the specified folder path.

        Parameters:
        -----------
        number_files_to_remove: int
            The number of files to remove from the folder path.

        Raises:
        -------
        IndexError:
            If the number of files to remove is greater than the number of
            files present in the folder, an IndexError is raised.

        Returns:
        --------
        None
        """

        # * Get a list of all files and directories in the specified folder path
        files = os.listdir(self.folder_path);

        # * Randomly select the specified number of files from the list
        for file_sample in sample(files, number_files_to_remove):
            os.remove(os.path.join(self.folder_path, file_sample))
            print(f"Removed {file_sample}")