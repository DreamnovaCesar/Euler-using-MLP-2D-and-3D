from abc import ABC
from abc import abstractmethod

from typing import List

class DataFrameCreator(ABC):

    @abstractmethod
    def create_dataframe_history(Column_names: list[str], Folder_save: str, CSV_name: str, Hist_data: object) -> None:
        pass