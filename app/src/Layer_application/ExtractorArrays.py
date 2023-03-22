import numpy as np

from .Extractor import Extractor
from ..Layer_domain.Arrays.ArraysHander import ArraysHandlder
from ..Layer_domain.DataLoaderText import DataLoaderText 

class ExtractorArrays(Extractor):
    """
    A class for extracting information from arrays.

    Parameters
    ----------
    Arrays_handler : ArraysHandler
        An arrays handler object that handles the arrays data.
    TextDataLoader : DataLoaderText
        A text data loader object that loads the text data.

    Attributes
    ----------
    Arrays_handler : ArraysHandler
        An arrays handler object that handles the arrays data.
    TextDataLoader : DataLoaderText
        A text data loader object that loads the text data.

    Methods
    -------
    extractor()
        Extracts information from the arrays.
    """
    
    def __init__(
        self, 
        Arrays_handler : ArraysHandlder, 
        TextDataLoader : DataLoaderText
    ):
        
        """
        Initializes the ExtractorArrays class.

        Parameters
        ----------
        Arrays_handler : ArraysHandler
            An arrays handler object that handles the arrays data.
        TextDataLoader : DataLoaderText
            A text data loader object that loads the text data.
        """

        # * Initialize the instance variables
        self.TextDataLoader = TextDataLoader
        self.Arrays_handler = Arrays_handler

    def extractor(
        self, 
        File : str
    ):
        
        """
        Extracts information from the arrays.

        Parameters
        ----------
        File : str
            The file path of the text data.

        Returns
        -------
        Arrays : ndarray
            An array of integers representing the arrays data.
        """
        
        # * Load data from the text file
        Arrays = self.TextDataLoader.load_data(File)

        # * Handle the arrays data
        Arrays_handler = self.Arrays_handler(Arrays)
        Arrays =  Arrays_handler.get_array()

        # * Print the resulting array
        print(Arrays)

        # * Return the resulting array
        return Arrays

    