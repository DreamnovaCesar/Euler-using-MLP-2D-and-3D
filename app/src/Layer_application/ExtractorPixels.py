# ? Utilities class for Euler number extraction for 2D and 3D
from .Extractor import Extractor
from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble
from ..Layer_domain.Arrays.PixelHander import PixelHandler
from ..Layer_domain.TextDataLoader import TextDataLoader 

class ExtractorPixels(Extractor):
    
    def __init__(self, Binary_storage_list : BinaryStorageList, 
                 Convertion_decimal_binary_nibble : ConvertionDecimalBinaryNibble, 
                 Pixel_handler : PixelHandler, 
                 TextDataLoader : TextDataLoader,
                 File : str):
        
        self.Binary_storage_list = Binary_storage_list
        self.Convertion_decimal_binary_nibble = Convertion_decimal_binary_nibble
        self.TextDataLoader = TextDataLoader
        self.Pixel_handler = Pixel_handler
        self.File = File

    def extractor(self):

        Binary_storage_list_object = self.Binary_storage_list(self.Convertion_decimal_binary_nibble)
        Storage_list = Binary_storage_list_object.to_numpy_array()

        Arrays = self.TextDataLoader.load_data(self.File)

        Pixel_handler_object = self.Pixel_handler(Arrays, Storage_list)
        Combinations_int = Pixel_handler_object.get_array()

        print(Combinations_int)