# ? Utilities class for Euler number extraction for 2D and 3D

from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble
from ..Layer_domain.Arrays.PixelHander import PixelHandler
from ..Layer_domain.TextDataLoader import TextDataLoader 

class ExtractorPixels(object):
    
    def __init__(self, Binary_storage_list : BinaryStorageList, 
                 Convertion_decimal_binary_nibble : ConvertionDecimalBinaryNibble, 
                 Pixel_handler : PixelHandler, 
                 TextDataLoader : TextDataLoader):
        
        self.Binary_storage_list = Binary_storage_list
        self.Convertion_decimal_binary_nibble = Convertion_decimal_binary_nibble
        self.TextDataLoader = TextDataLoader
        self.Pixel_handler = Pixel_handler

    def extractor(self):

        Binary_storage_list_object = self.Binary_storage_list(self.Convertion_decimal_binary_nibble)
        Qs = Binary_storage_list_object.to_numpy_array()

        Arrays = self.TextDataLoader.load_data(r'app\Data\3D\Images backup\Image_random_0_3D.txt')

        Pixel_handler_object = self.Pixel_handler(Arrays, Qs)
        q_values = Pixel_handler_object.get_number()

        print(q_values)