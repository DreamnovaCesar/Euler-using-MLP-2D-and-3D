
import pandas as pd

from .DataLoader import DataLoader

class CSVReader(DataLoader):

    def load_data(file_path):
        return pd.read_csv(file_path)