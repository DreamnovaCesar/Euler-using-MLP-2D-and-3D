import numpy as np

import joblib
from keras.models import load_model

from keras.models import Sequential

from ..Layer_domain.Arrays.ArraysHander import ArraysHandlder
from ..Layer_domain.DataLoaderText import DataLoaderText 
from ..Layer_domain.Model.MLP import MLP

from .ExtractorArrays import ExtractorArrays

from .MLPPrediction import MLPPrediction

class MLPPredictionStandard(MLPPrediction):
    """
    A class for making predictions using an MLP model trained on standard arrays.

    Attributes
    ----------
    Extractor_arrays : ExtractorArrays
        An object for extracting arrays from data.
    MLP_prediction : MLP
        An MLP model for making predictions.

    Methods
    -------
    prediction(Model, Object):
        Predicts the output of an input object using an MLP model.

    """

    def __init__(
        self, 
        Extractor_arrays : ExtractorArrays,
        MLP_prediction : MLP
    ):
        """
        Initializes the MLPPredictionStandard class.

        Parameters
        ----------
        Extractor_arrays : ExtractorArrays
            An object for extracting arrays from data.
        MLP_prediction : MLP
            An MLP model for making predictions.

        """

        self.Extractor_arrays = Extractor_arrays
        self.MLP_prediction = MLP_prediction

    def prediction(
        self, 
        Model : Sequential, 
        Object : str
    ):

        """
        Predicts the output of an input object using an MLP model.

        Parameters
        ----------
        Model : Sequential
            The trained MLP model.
        Object : str
            The input object for prediction.

        Returns
        -------
        Prediction_result : float
            The predicted result.

        """

        # * Extract the array from the object
        Arrays = self.Extractor_arrays(
            ArraysHandlder,
            DataLoaderText
        )

        Arrays = Arrays.extractor(Object)

        # * Initialize the prediction result to zero
        Prediction_result = 0

        # * Read multilayer perceptron model
        if Model.endswith('.joblib'):
            
            Model_prediction = joblib.load(Model)

            # * Loop through the input array
            for _, Array in enumerate(Arrays):

                Result = np.argmax(
                    self.MLP_prediction.predict_model(
                        Model_prediction, 
                        [Array]), 
                    axis = 1
                );

                print(Result[0])

                # * Map the integer output to corresponding values
                Result_map = {0: 0, 1: 1, 2: -1, 3: -2};
                Result = Result_map.get(Result[0]);

                # * Print the prediction and true values
                print('{} -------------- {}'.format(Array, Result));
                print('The result is: {}'.format(Result));
                print('The true value is: {}'.format(Result));
                print('\n');

                # * Add current prediction result to the overall prediction result
                Prediction_result += Result;

                # * Print the final prediction result
                print(Prediction_result);
        
        return Prediction_result