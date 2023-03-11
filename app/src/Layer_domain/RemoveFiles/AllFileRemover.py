import os
import shutil

from .FileRemover import FileRemover

class AllFileRemover(FileRemover):
    """
    A class that inherits from the `FileRemover` class and implements
    the `remove_files` method to remove all files and directories
    from a specified folder path.

    Attributes:
    -----------
    folder_path: str
        The path of the folder to remove all files and directories from.

    Methods:
    --------
    remove_files() -> None:
        Removes all files and directories from the specified folder path.
        Raises an exception if the removal process fails for any file/directory.

    """
    
    def remove_files(self) -> None:
        """
        Removes all files and directories from the specified folder path.

        Raises:
        -------
        Exception:
            If the removal process fails for any file/directory, the exception
            is raised with the reason for failure printed to the console.

        Returns:
        --------
        None
        """

        # * Loop through all files and directories in the specified folder path
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename);
            try:
                # * If the path is a file or a symbolic link, delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path);
                # * If the path is a directory, remove it and its contents
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path);
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")