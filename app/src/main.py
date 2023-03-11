from .Layer_application.ExtractorPixels import ExtractorPixels
from .Layer_application.ExtractorOctovoxels import ExtractorOctovoxels

from .Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from .Layer_domain.Convertion.ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from .Layer_domain.Convertion.ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble
from .Layer_domain.Arrays.OctovoxelHander import OctovoxelHandler
from .Layer_domain.Arrays.PixelHander import PixelHandler
from .Layer_domain.TextDataLoader import TextDataLoader 


class TODO:

    EEO = ExtractorOctovoxels(BinaryStorageList, ConvertionDecimalBinaryByte, 
                                OctovoxelHandler, TextDataLoader)

    EEO.extractor()

    EEO = ExtractorPixels(BinaryStorageList, ConvertionDecimalBinaryNibble, 
                            PixelHandler, TextDataLoader)

    EEO.extractor()