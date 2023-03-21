import numpy as np

from keras.models import load_model

from ..Layer_domain.Arrays.ArraysHander import ArraysHandlder
from ..Layer_domain.DataLoaderText import DataLoaderText 
from ..Layer_domain.Model.MLP import MLP

from .ExtractorArrays import ExtractorArrays

from .MLPPrediction import MLPPrediction

class MLPPredictionStandard(MLPPrediction):
    
    def __init__(self, 
                 Extractor_arrays : ExtractorArrays,
                 MLP_prediction : MLP):
        
        self.Extractor_arrays = Extractor_arrays
        self.MLP_prediction = MLP_prediction

    def prediction(self, Model, Object):

        Arrays = self.Extractor_arrays(ArraysHandlder,
                                       DataLoaderText,
                                       Object)

        Arrays = Arrays.extractor()

        # * Initialize the prediction result to zero
        Prediction_result = 0

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):
            
            Model_prediction = load_model(Model)

            # * Loop through the input array
            for i, Array in enumerate(Arrays):

                Result = np.argmax(self.MLP_prediction.predict_model(Model_prediction, 
                                                                    [Array]), axis = 1)

                #print(Result)
                print(i)

                if(Result == 0):
                    Result = 0
                elif(Result == 1):
                    Result = 1
                elif(Result == 2):
                    Result = -1
                elif(Result == 3):
                    Result = -2

                # * Print the prediction and true values
                print('{} -------------- {}'.format(Array, Result))
                print('The result is: {}'.format(Result))
                print('The true value is: {}'.format(Result))
                print('\n')

                # * Print the final prediction result
                Prediction_result += Result

                print(Prediction_result)

        return Prediction_result