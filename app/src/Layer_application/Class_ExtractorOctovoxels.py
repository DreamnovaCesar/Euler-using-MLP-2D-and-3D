
from ..Layer_domain.Convertion.Class_BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.Class_ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from ..Layer_domain.getArray.Class_OctovoxelHander import OctovoxelHandler
from ..Layer_domain.Class_TextDataLoader import TextDataLoader 

class ExtractorOctovoxels(object):
    
    def __init__(self, Binary_storage_list : BinaryStorageList, 
                 Convertion_decimal_binary_byte : ConvertionDecimalBinaryByte, 
                 Octovoxel_handler : OctovoxelHandler, 
                 TextDataLoader : TextDataLoader):
        
        self.Binary_storage_list = Binary_storage_list
        self.Convertion_decimal_binary_byte = Convertion_decimal_binary_byte
        self.TextDataLoader = TextDataLoader
        self.Octovoxel_handler = Octovoxel_handler

    def extractor(self):

        Binary_storage_list_object = self.Binary_storage_list(self.Convertion_decimal_binary_byte)
        Qs = Binary_storage_list_object.to_numpy_array()
        Arrays = self.TextDataLoader.load_data(r'app\Data\3D\Images backup\Image_random_0_3D.txt')
        Octovoxel_handler_object = self.Octovoxel_handler(Arrays, Qs)
        q_values = Octovoxel_handler_object.get_number()

        print(q_values)

    