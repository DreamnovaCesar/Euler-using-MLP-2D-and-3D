from Article_Euler_Number_Libraries import *
from Article_Euler_Number_Utilities import Utilities

class RemoveFiles(Utilities):

    def __init__(self, **kwargs) -> None:

        # * Instance attributes (Protected)
        self.__Folder_path = kwargs.get('folder', None);
        self.__Number_Files_to_remove = kwargs.get('NFR', None);

    def __repr__(self):

        kwargs_info = "[{}, {}]".format(self.__Folder_path, self.__Number_Files_to_remove);

        return kwargs_info

    def __str__(self):
        pass
    
    # * Folder_path attribute
    @property
    def Folder_path_property(self):
        return self.__Folder_path

    @Folder_path_property.setter
    def Folder_path_property(self, New_value):
        if not isinstance(New_value, str):
            raise TypeError("Folder_path must be a string") #! Alert
        self.__Folder_path = New_value;
    
    @Folder_path_property.deleter
    def Folder_path_property(self):
        print("Deleting Folder_path...");
        del self.__Folder_path

    # * Files_to_remove attribute
    @property
    def Files_to_remove_property(self):
        return self.__Files_to_remove

    @Files_to_remove_property.setter
    def Files_to_remove_property(self, New_value):
        if not isinstance(New_value, int):
            raise TypeError("Files_to_remove must be a integer") #! Alert
        self.__Files_to_remove = New_value;
    
    @Files_to_remove_property.deleter
    def Files_to_remove_property(self):
        print("Deleting Files_to_remove...");
        del self.__Files_to_remove

    # ? Remove all files inside a dir
    @Utilities.time_func
    def remove_all_files(self) -> None:
        """
        Remove all files inside the folder path obtained.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            None
        """
        
        # * Folder attribute (ValueError, TypeError)
        if self.__Folder_path == None:
            raise ValueError("Folder does not exist") #! Alert
        if not isinstance(self.__Folder_path, str):
            raise TypeError("Folder must be a string") #! Alert

        # * This function will remove all the files inside a folder
        for File in os.listdir(self.__Folder_path):
            Filename, Format  = os.path.splitext(File);
            print('Removing: {} . {} ✅'.format(Filename, Format));
            os.remove(os.path.join(self.__Folder_path, File));

    @Utilities.time_func
    def remove_random_files(self) -> None:
        """
        Remove all files inside the folder path obtained.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            None
        """

        # * This function will remove all the files inside a folder
        Files = os.listdir(self.__Folder_path);

            #Filename, Format = os.path.splitext(File)

        for File_sample in sample(Files, self.__Number_Files_to_remove):
            print(File_sample);
            #print('Removing: {}{} ✅'.format(Filename, Format));
            os.remove(os.path.join(self.__Folder_path, File_sample));

    # ? Remove all files inside a dir
    @Utilities.time_func
    def remove_all(self) -> None:
        """
        Remove all files inside the folder path obtained.

        Args:
            Folder_path (str): Folder path obtained.

        Returns:
            None
        """
        
        for Filename in os.listdir(self.__Folder_path):

            File_path = os.path.join(self.__Folder_path, Filename)

            try:
                if os.path.isfile(File_path) or os.path.islink(File_path):
                    os.unlink(File_path)
                elif os.path.isdir(File_path):
                    shutil.rmtree(File_path)

            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (File_path, e))