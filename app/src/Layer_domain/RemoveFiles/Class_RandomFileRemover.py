import os
from random import sample

from .Class_FileRemover import FileRemover

class RandomFileRemover(FileRemover):
    def remove_files(self, number_files_to_remove: int) -> None:
        files = os.listdir(self.folder_path)
        for file_sample in sample(files, number_files_to_remove):
            os.remove(os.path.join(self.folder_path, file_sample))
            print(f"Removed {file_sample}")