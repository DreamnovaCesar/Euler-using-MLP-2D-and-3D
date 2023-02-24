# ? Utilities class for Euler number extraction for 2D and 3D

from .Class_EulerExtractor import EulerExtractor

class EulerExtractorImage(EulerExtractor):
    
    def __init__(self, data_reader, initial_x, final_x):
        self.data_reader = data_reader
        self.initial_x = initial_x
        self.final_x = final_x
        
    def euler_extractor():
        pass