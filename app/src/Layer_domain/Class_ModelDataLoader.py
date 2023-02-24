
from keras.models import load_model

from .Class_DataLoader import DataLoader

class DataLoaderModel(DataLoader):

    def load_data(Model):
        return load_model(Model)