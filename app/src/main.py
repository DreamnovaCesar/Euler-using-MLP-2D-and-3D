from .Layer_application.Class_ExtractorPixels import ExtractorPixels
from .Layer_application.Class_ExtractorOctovoxels import ExtractorOctovoxels

from .Layer_domain.Convertion.Class_BinaryStorageList import BinaryStorageList
from .Layer_domain.Convertion.Class_ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from .Layer_domain.Convertion.Class_ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble
from .Layer_domain.getArray.Class_OctovoxelHander import OctovoxelHandler
from .Layer_domain.getArray.Class_PixelHander import PixelHandler
from .Layer_domain.Class_TextDataLoader import TextDataLoader 


class TODO:

    EEO = ExtractorOctovoxels(BinaryStorageList, ConvertionDecimalBinaryByte, 
                                OctovoxelHandler, TextDataLoader)

    EEO.extractor()

    EEO = ExtractorPixels(BinaryStorageList, ConvertionDecimalBinaryNibble, 
                            PixelHandler, TextDataLoader)

    EEO.extractor()