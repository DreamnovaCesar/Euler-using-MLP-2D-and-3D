import os

class FileSaver:
    def __init__(self, folder_data):
        self._Folder_data = folder_data

    def save_file(self, file_name):
        return os.path.join(self._Folder_data, file_name)