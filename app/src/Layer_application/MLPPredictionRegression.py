from keras.models import load_model

from keras.models import Sequential

from ..Layer_domain.Convertion.BinaryStorageList import BinaryStorageList
from ..Layer_domain.Convertion.ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from ..Layer_domain.Arrays.OctovoxelHander import OctovoxelHandler
from ..Layer_domain.DataLoaderText import DataLoaderText
from ..Layer_domain.Model.MLP import MLP

from .ExtractorOctovoxels import ExtractorOctovoxels

from .MLPPrediction import MLPPrediction

class MLPPredictionRegression(MLPPrediction):
    """
    A class for performing multilayer perceptron (MLP) predictions with regression outputs.

    Attributes
    ----------
    Extractor_octovoxel : ExtractorOctovoxels
        An object used for extracting octovoxel data from input objects.
    MLP_prediction : MLP
        An object used for making MLP predictions.
    Octovoxels : OctovoxelHandler
        An object used for handling octovoxel data.
    
    Methods
    -------
    prediction(Model, Object)
        Predicts the output of the given object using the specified MLP model.

    """

    def __init__(
        self, 
        Extractor_octovoxel : ExtractorOctovoxels,
        MLP_prediction : MLP
    ):
        """
        Initializes the MLPPredictionRegression object.

        Parameters
        ----------
        Extractor_octovoxel : ExtractorOctovoxels
            An object used for extracting octovoxel data from input objects.
        MLP_prediction : MLP
            An object used for making MLP predictions.

        Returns
        -------
        None

        """

        self.Extractor_octovoxel = Extractor_octovoxel
        self.MLP_prediction = MLP_prediction

        self.Octovoxels = self.Extractor_octovoxel(
            BinaryStorageList,
            ConvertionDecimalBinaryByte,
            OctovoxelHandler,
            DataLoaderText
        )
        
    def prediction(
        self, 
        Model : Sequential, 
        Object : str
    ):

        """
        Predicts the output of the given object using the specified MLP model.

        Parameters
        ----------
        Model : str
            The path to the saved MLP model.
        Object : str
            The object to predict the output of.

        Returns
        -------
        None

        """
        Combinations_int = self.Octovoxels.extractor(Object)

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):

            Model_prediction = load_model(Model)
            Result = self.MLP_prediction.predict_model(
                Model_prediction, 
                Combinations_int
            );

            print(Result);

        return Result

