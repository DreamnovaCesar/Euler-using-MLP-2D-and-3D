import json
from .Class_JsonOperations import JsonOperations

class JsonFileHandler(JsonOperations):
    def create_json_file(self, data: dict, file_path: str) -> None:
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def read_json_file(self, file_path: str) -> dict:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data