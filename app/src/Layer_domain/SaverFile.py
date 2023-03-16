import os

from .Saver import Saver

class SaverFile(Saver):
    def __init__(self, Folder_data):
        self._Folder_data = Folder_data

    def save_file(self, File_name):
        return os.path.join(self._Folder_data, File_name)