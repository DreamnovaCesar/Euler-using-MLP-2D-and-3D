from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities

class RemoveFiles(object):
    """
    A class used to remove files inside a folder.

    Parameters
    ----------
    folder : str, optional
        Path to the folder containing files to be removed, by default None.
    NFR : int, optional
        Number of files to be removed, by default None.

    Attributes
    ----------
    Folder_path_property : str
        Path to the folder containing files to be removed.
    Files_to_remove_property : int
        Number of files to be removed.

    Methods
    -------
    remove_all_files()
        Remove all the files inside the folder.
    remove_random_files()
        Remove a random number of files inside the folder.
    remove_all()
        Remove all the files and directories inside the folder.

    Raises
    ------
    ValueError
        If the folder does not exist.
    TypeError
        If the folder or number of files is not a string or integer respectively.

    Notes
    -----
    The `remove_all_files()` method removes all the files inside the specified folder.
    The `remove_random_files()` method removes a random number of files inside the specified folder.
    The `remove_all()` method removes all the files and directories inside the specified folder.

    Examples
    --------
    >>> r = RemoveFiles(folder='/path/to/folder', NFR=5)
    >>> r.remove_all_files()
    Removing: file1 . txt ✅
    Removing: file2 . txt ✅
    Removing: file3 . txt ✅
    Removing: file4 . txt ✅
    Removing: file5 . txt ✅
    >>> r.remove_random_files()
    file6.txt
    file7.txt
    file8.txt
    >>> r.remove_all()
    """
    
    # * Initializing (Constructor)
    def __init__(self, **kwargs) -> None:
        """
        Constructor method for RemoveFiles.

        Keyword Arguments
        -----------------
        Folder : str
            Folder intended for removing files inside it

        """
        # * Instance attributes (Protected)
        self.__Folder = kwargs.get('Folder', None);

    # * Class variables
    def __repr__(self):
        """
        Returns the string representation of this RemoveFiles instance.

        Returns
        -------
        str 
            The string representation of this RemoveFiles instance.
        """
        return f'''[{self.__Folder}]''';

    # * Class description
    def __str__(self):
        """
        Returns a string representation of this RemoveFiles instance.

        Returns
        -------
        str 
            The string representation of this RemoveFiles instance.
        """
        return  f'A class used to remove files inside a folder.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor that is called when this RemoveFiles instance is destroyed.
        """
        print('Destructor called, RemoveFiles class destroyed.');

    # * Get data from a dic
    def data_dic(self):

        return {'Folder': str(self.__Folder)
                };
    
    # * __Folder attribute
    @property
    def __Folder_property(self):
        """Getter method for the `Folder` property."""
        return self.__Folder

    @__Folder_property.setter
    def __Folder_property(self, New_value):
        """Setter method for the `Folder` property.

        Parameters
        ----------
        New_value : str
            The new value to be assigned to the `Folder` attribute.
        """
        self.__Folder = New_value;
    
    @__Folder_property.deleter
    def __Folder_property(self):
        """Deleter method for the `Folder` property."""
        print("Deleting Folder...");
        del self.__Folder

    # ? Method to remove all the files inside the dir
    def remove_all_files(self) -> None:
        """
        Remove all the files inside the directory.

        This method removes all the files located inside the directory specified by 
        the `__Folder_path` attribute.

        """

        # * Loop through all files in the folder
        for File in os.listdir(self.__Folder):
            # * Split the file name into filename and extension
            Filename, Format  = os.path.splitext(File);
            # * Print a message indicating that the file is being removed
            print('Removing: {} . {} ✅'.format(Filename, Format));
            os.remove(os.path.join(self.__Folder, File));

    # ? Method to files randomly inside a dir
    def remove_random_files(self, Number_Files_to_remove) -> None:
        """
        Remove files randomly inside the folder path.

        This method removes a specified number of files randomly from the directory
        specified by the `__Folder_path` attribute.

        Parameters
        ----------
        Number_Files_to_remove : int
            The number of files to remove randomly from the directory.

        """

        # * Get a list of all files in the folder
        Files = os.listdir(self.__Folder);

        # * Choose a random sample of files to remove
        for File_sample in sample(Files, Number_Files_to_remove):
            # * Print the name of the file being removed
            os.remove(os.path.join(self.__Folder, File_sample));
            print(File_sample);

    # ? 
    def remove_all(self) -> None:
        """
        Remove all the files and directories inside the folder path.

        This method removes all the files and directories located inside the 
        directory specified by the `__Folder_path` attribute.

        """

        # * Loop through all files and directories in the folder
        for Filename in os.listdir(self.__Folder):
                
            # * Get the full path of the file or directory
            File_path = os.path.join(self.__Folder, Filename);

            try:
                # * If the path is a file or a symbolic link, remove it
                if os.path.isfile(File_path) or os.path.islink(File_path):
                    os.unlink(File_path);
                # * If the path is a directory, remove it and all its contents
                elif os.path.isdir(File_path):
                    shutil.rmtree(File_path);

            except Exception as e:
                # * Print an error message if the file or directory could not be deleted
                print('Failed to delete %s. Reason: %s' % (File_path, e));