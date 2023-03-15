import numpy as np

from .Extractor import Extractor
from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from ..Layer_domain.Arrays.OctovoxelHander import OctovoxelHandler
from ..Layer_domain.TextDataLoader import TextDataLoader 

class ExtractorOctovoxels(Extractor):
    
    def __init__(self, Binary_storage_list : BinaryStorageList, 
                 Convertion_decimal_binary_byte : ConvertionDecimalBinaryByte, 
                 Octovoxel_handler : OctovoxelHandler, 
                 TextDataLoader : TextDataLoader
                 ):
        
        self.Binary_storage_list = Binary_storage_list
        self.Convertion_decimal_binary_byte = Convertion_decimal_binary_byte
        self.TextDataLoader = TextDataLoader
        self.Octovoxel_handler = Octovoxel_handler

    def extractor(self, File):

        Binary_storage_list_object = self.Binary_storage_list(self.Convertion_decimal_binary_byte)
        Storage_list = Binary_storage_list_object.to_numpy_array()

        Arrays = self.TextDataLoader.load_data(File)

        Octovoxel_handler_object = self.Octovoxel_handler(Arrays, Storage_list)
        Combinations_int = Octovoxel_handler_object.get_array()

        # *
        Combinations_int = np.expand_dims(Combinations_int, axis = 0)

        print(Combinations_int)

        return Combinations_int

    