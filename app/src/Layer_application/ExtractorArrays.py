import numpy as np

from .Extractor import Extractor
from ..Layer_domain.Arrays.ArraysHander import ArraysHandlder
from ..Layer_domain.TextDataLoader import TextDataLoader 

class ExtractorArrays(Extractor):
    
    def __init__(self, 
                 Arrays_handler : ArraysHandlder, 
                 TextDataLoader : TextDataLoader,
                 File : str):
        
        self.TextDataLoader = TextDataLoader
        self.Arrays_handler = Arrays_handler
        self.File = File

    def extractor(self):

        Arrays = self.TextDataLoader.load_data(self.File)

        Arrays_handler = self.Arrays_handler(Arrays)
        Arrays =  Arrays_handler.get_array()

        print(Arrays)

        return Arrays

    