import numpy as np 

from .DataLoaderCSV import DataLoaderCSV

class DataProcessor(object):
    """
    A class that processes input data and labels.

    Parameters:
    -----------
    Data_reader : DataLoaderCSV
        An instance of the DataLoaderCSV class used to read data.
    Initial_x : int, optional (default=1)
        The starting column index of input data in the data file.
    Final_x : int, optional (default=257)
        The last column index of input data in the data file.

    Attributes:
    -----------
    Data_reader : DataLoaderCSV
        The instance of the DataLoaderCSV class used to read data.
    Initial_x : int
        The starting column index of input data in the data file.
    Final_x : int
        The last column index of input data in the data file.
    
    """

    def __init__(self, Data_reader : DataLoaderCSV, 
                 Initial_x : int = 1, 
                 Final_x : int = 257):
        
        self.Data_reader = Data_reader
        self.Initial_x = Initial_x
        self.Final_x = Final_x

    @property
    def process_data(
        self, 
        Path : str
    ):
        """
        Read input data and labels from a file, and reshape the labels array.

        Parameters:
        -----------
        Path : str
            The path to the file containing the input data and labels.

        Returns:
        --------
        X : numpy.ndarray
            A two-dimensional numpy array of input data.
        Y : numpy.ndarray
            A two-dimensional numpy array of labels, reshaped to have the same number of dimensions as the input data.
        """

        # * read the data from the source using the data reader
        dataframe = self.Data_reader.load_data(Path)

        # * extract the input data and labels from the dataframe
        X = dataframe.iloc[:, self.Initial_x:self.Final_x].values
        Y = dataframe.iloc[:, -1].values

        # * reshape the labels array to have the same number of dimensions as the input data
        Y = np.expand_dims(Y, axis = 1)

        return X, Y