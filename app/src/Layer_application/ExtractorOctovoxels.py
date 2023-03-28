import numpy as np

from .Extractor import Extractor
from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from ..Layer_domain.Arrays.OctovoxelHander import OctovoxelHandler
from ..Layer_domain.DataLoaderText import DataLoaderText  

class ExtractorOctovoxels(Extractor):
    """
    A class for extracting information from octovoxels data.

    Attributes
    ----------
    Binary_storage_list : BinaryStorageList
        A binary storage list object that contains the binary storage list.
    Convertion_decimal_binary_byte : ConvertionDecimalBinaryByte
        A conversion object that converts decimal to binary byte.
    Octovoxel_handler : OctovoxelHandler
        An octovoxel handler object that handles the octovoxel data.
    TextDataLoader : DataLoaderText
        A text data loader object that loads the text data.

    Methods
    -------
    extractor():
        Extracts information from the octovoxels data in the file.
    """

    def __init__(
        self, 
        Binary_storage_list : BinaryStorageList, 
        Convertion_decimal_binary_byte : ConvertionDecimalBinaryByte, 
        Octovoxel_handler : OctovoxelHandler, 
        TextDataLoader : DataLoaderText 
    ):
        
        """
        Initializes the ExtractorOctovoxels instance.

        Parameters
        ----------
        Binary_storage_list : BinaryStorageList
            A binary storage list object that contains the binary storage list.
        Convertion_decimal_binary_byte : ConvertionDecimalBinaryByte
            A conversion object that converts decimal to binary byte.
        Octovoxel_handler : OctovoxelHandler
            An octovoxel handler object that handles the octovoxel data.
        TextDataLoader : DataLoaderText
            A text data loader object that loads the text data.
        """

        # * Initialize the instance variables
        self.Binary_storage_list = Binary_storage_list;
        self.Convertion_decimal_binary_byte = Convertion_decimal_binary_byte;
        self.TextDataLoader = TextDataLoader;
        self.Octovoxel_handler = Octovoxel_handler;

    def extractor(
        self, 
        File : str
    ):
        
        """
        Extracts information from the octovoxels.

        Parameters
        ----------
        File : str
            The file path of the text data.

        Returns
        -------
        Combinations_int : ndarray
            An array of integers representing the octovoxel data combinations.
        """

        # * Create a BinaryStorageList object with the Convertion_decimal_binary_byte object and convert it to a numpy array
        Binary_storage_list_object = self.Binary_storage_list(self.Convertion_decimal_binary_byte);
        Storage_list = Binary_storage_list_object.to_numpy_array();

        # * Load the text data from the file using the TextDataLoader object
        Arrays = self.TextDataLoader.load_data(File);

        # * Create an OctovoxelHandler object with the Arrays and Storage_list objects and get the octovoxel data combinations
        Octovoxel_handler_object = self.Octovoxel_handler(Arrays, Storage_list);
        Combinations_int = Octovoxel_handler_object.get_array();

        # * Expand the dimensions of Combinations_int to match the expected output format
        Combinations_int = np.expand_dims(Combinations_int, axis = 0);

        # * Print Combinations_int for debugging purposes
        print(Combinations_int);

        return Combinations_int

    