
import pandas as pd

from .DataLoader import DataLoader

class DataLoaderCSV(DataLoader):
    """
    A class that inherits from the `DataLoader` base class and loads data from a CSV file.

    Methods
    -------
    load_data(file_path)
        Load the data from the CSV file specified by the file path.

    """
    def load_data(File_path : str):
        """
        Load data from a CSV file.

        Parameters
        ----------
        file_path : str
            The path of the CSV file to be loaded.

        Returns
        -------
        pandas.DataFrame
            The data loaded from the CSV file as a pandas DataFrame.
        """
        return pd.read_csv(File_path)