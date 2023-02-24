# ? Utilities class for Euler number extraction for 2D and 3D

from .Class_EulerExtractor import EulerExtractor

class EulerExtractorObject(EulerExtractor):
    
    def __init__(self, Binary_storage_list : object, Convertion_decimal_binary_byte : object, 
                 Octovoxel_handler : object, DataLoaderTxt : object):
        
        self.Binary_storage_list = Binary_storage_list
        self.Convertion_decimal_binary_byte = Convertion_decimal_binary_byte
        self.DataLoaderTxt = DataLoaderTxt
        self.Octovoxel_handler = Octovoxel_handler

    def euler_extractor(self):

        Binary_storage_list_object = self.Binary_storage_list(self.Convertion_decimal_binary_byte)
        Qs = Binary_storage_list_object.to_numpy_array()
        Arrays = self.DataLoaderTxt.load_data(r'app\Data\3D\Images backup\Image_random_0_3D.txt')
        Octovoxel_handler_object = self.Octovoxel_handler(Arrays)
        q_values = Octovoxel_handler_object.get_number(Qs)

        print(q_values)
