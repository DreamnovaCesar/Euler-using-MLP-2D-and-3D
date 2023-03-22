# ? Utilities class for Euler number extraction for 2D and 3D
from .Extractor import Extractor
from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble
from ..Layer_domain.Arrays.PixelHander import PixelHandler
from ..Layer_domain.DataLoaderText import DataLoaderText 

class ExtractorPixels(Extractor):
    """
    A class that extracts information from pixels.

    Parameters
    ----------
    Binary_storage_list : BinaryStorageList
        A binary storage list object that contains the binary storage list.
    Convertion_decimal_binary_nibble : ConvertionDecimalBinaryNibble
        A conversion object that converts decimal to binary nibble.
    Pixel_handler : PixelHandler
        A pixel handler object that handles the pixel data.
    TextDataLoader : DataLoaderText
        A text data loader object that loads the text data.
    File : str
        The file path of the text data.

    Attributes
    ----------
    Binary_storage_list : BinaryStorageList
        The binary storage list object.
    Convertion_decimal_binary_nibble : ConvertionDecimalBinaryNibble
        The conversion object.
    TextDataLoader : DataLoaderText
        The text data loader object.
    Pixel_handler : PixelHandler
        The pixel handler object.
    File : str
        The file path of the text data.

    Methods
    -------
    extractor()
        Extracts information from the pixels.
    """

    def __init__(
        self, 
        Binary_storage_list : BinaryStorageList, 
        Convertion_decimal_binary_nibble : ConvertionDecimalBinaryNibble, 
        Pixel_handler : PixelHandler, 
        TextDataLoader : DataLoaderText
    ):
        
        """
        Initializes the ExtractorPixels class.

        Parameters
        ----------
        Binary_storage_list : BinaryStorageList
            A binary storage list object that contains the binary storage list.
        Convertion_decimal_binary_nibble : ConvertionDecimalBinaryNibble
            A conversion object that converts decimal to binary nibble.
        Pixel_handler : PixelHandler
            A pixel handler object that handles the pixel data.
        TextDataLoader : DataLoaderText
            A text data loader object that loads the text data.
        """

        # * Initialize the instance variables
        self.Binary_storage_list = Binary_storage_list
        self.Convertion_decimal_binary_nibble = Convertion_decimal_binary_nibble
        self.TextDataLoader = TextDataLoader
        self.Pixel_handler = Pixel_handler

    def extractor(
        self, 
        File : str
    ):

        """
        Extracts information from the pixels.
        
        Parameters
        ----------
        File : str
            The file path of the text data.

        Returns
        -------
        Combinations_int : ndarray
            An array of integers representing the octovoxel data combinations.
        """

        # * Creates a BinaryStorageList object and calls the to_numpy_array method to get the binary storage list as a NumPy array
        Binary_storage_list = self.Binary_storage_list(self.Convertion_decimal_binary_nibble)
        Storage_list = Binary_storage_list.to_numpy_array()

        # * Loads the text data from the file path
        Arrays = self.TextDataLoader.load_data(File)

        # * Creates a PixelHandler object and calls the get_array method to get the pixel data as an array of integers
        Pixel_handler = self.Pixel_handler(Arrays, Storage_list)
        Combinations_int = Pixel_handler.get_array()

        # * Prints the array of integers representing the pixel data combinations
        print(Combinations_int)
        
        return Combinations_int