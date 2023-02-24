import json
from abc import ABC, abstractmethod

class JsonOperations(ABC):
    @abstractmethod
    def create_json_file(self, data: dict, file_path: str) -> None:
        pass

    @abstractmethod
    def read_json_file(self, file_path: str) -> dict:
        pass