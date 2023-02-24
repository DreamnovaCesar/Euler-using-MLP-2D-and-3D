import numpy as np

from .Class_DataLoader import DataLoader

class TextDataLoader (DataLoader):
    """
    A class that loads data from files.
    """
    
    def load_data(file_path):
        """
        Load data from a file.

        Parameters
        ----------
        file_path : str
            The path to the input data file.

        Returns
        -------
        np.ndarray
            The input data as a numpy array.
        """
        return np.loadtxt(file_path, delimiter=',')