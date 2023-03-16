import numpy as np

from .DataLoader import DataLoader

class DataLoaderText(DataLoader):
    """
    A class that loads data from files.
    """
    
    def load_data(File_path):
        """
        Load data from a file.

        Parameters
        ----------
        File_path : str
            The path to the input data file.

        Returns
        -------
        np.ndarray
            The input data as a numpy array.
        """
        return np.loadtxt(File_path, delimiter=',')