from abc import ABC
from abc import abstractmethod

class FileRemover(ABC):
    def __init__(self, folder_path: str) -> None:
        self.folder_path = folder_path

    @abstractmethod
    def remove_files(self) -> None:
        pass