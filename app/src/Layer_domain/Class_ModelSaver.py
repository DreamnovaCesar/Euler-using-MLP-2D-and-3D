from abc import ABC
from abc import abstractmethod

class ModelSaver(ABC):
    def __init__(self, file_saver):
        self.file_saver = file_saver

    def save_model(self, model, model_name):
        pass