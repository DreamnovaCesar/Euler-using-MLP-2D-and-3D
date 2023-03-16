from keras.models import load_model

from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from ..Layer_domain.Arrays.OctovoxelHander import OctovoxelHandler
from ..Layer_domain.DataLoaderText import TextDataLoader 
from ..Layer_domain.Model.MLP import MLP

from .ExtractorOctovoxels import ExtractorOctovoxels

from .MLPPrediction import MLPPrediction

class MLPPredictionRegression(MLPPrediction):
    
    def __init__(self, 
                 Extractor_octovoxel : ExtractorOctovoxels,
                 MLP_prediction : MLP):
        
        self.Extractor_octovoxel = Extractor_octovoxel
        self.MLP_prediction = MLP_prediction

        self.Octovoxels = self.Extractor_octovoxel(BinaryStorageList,
                                                   ConvertionDecimalBinaryByte,
                                                   OctovoxelHandler,
                                                   TextDataLoader
                                                   )
        
    def prediction(self, Model, Object):

        Combinations_int = self.Octovoxels.extractor(Object)

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):

            Model_prediction = load_model(Model)
            Result = self.MLP_prediction.predict_model(Model_prediction, 
                                                       Combinations_int)
            print(Result)

