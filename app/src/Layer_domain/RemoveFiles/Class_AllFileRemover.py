import os
import shutil

from .Class_FileRemover import FileRemover

class AllFileRemover(FileRemover):
    def remove_files(self) -> None:
        for filename in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")