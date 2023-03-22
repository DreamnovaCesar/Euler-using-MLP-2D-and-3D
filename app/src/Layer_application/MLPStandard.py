import numpy as np
from ..Layer_domain.Model.MLP import MLP

from ..Layer_domain.Model.ModelSaverDL import ModelSaverDL
from ..Layer_presentation.DataPlotterDL import DataPlotterDL
from ..Layer_domain.DataFrameCreatorDL import DataFrameCreatorDL

from .MLPTrain import MLPTrain

class MLPStandard(MLPTrain):
    """
    A class to train a multilayer perceptron (MLP) model using the standard training algorithm.

    Parameters
    ----------
    MLP_training : MLP
        A multilayer perceptron (MLP) model to be trained.

    Methods
    -------
    train(X, Y, JSON_file, Model_name, epochs=10000)
        Trains the MLP model.

    Attributes
    ----------
    MLP_training : MLP
        A multilayer perceptron (MLP) model to be trained.

    """
    def __init__(self, 
                 MLP_training : MLP):
        """
        Initializes the MLPStandard class.

        Parameters
        ----------
        MLP_training : MLP
            A multilayer perceptron (MLP) model to be trained.
        """

        self.MLP_training = MLP_training

    def train(
        self, 
        X : np.ndarray, 
        Y : np.ndarray, 
        JSON_file : str, 
        Model_name : str,
        Epochs : int = 10000, 
    ):
        
        """
        Trains the multilayer perceptron (MLP) model using the standard training algorithm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data.
        Y : array-like of shape (n_samples,)
            The target values.
        JSON_file : str
            The file path to the JSON file containing the model's hyperparameters.
        Model_name : str
            The name of the model.
        epochs : int, optional
            The number of training epochs. Default is 10000.

        Returns
        -------
        None

        """

        # * Instantiates an MLP object to train
        MLP = self.MLP_training(
            X, 
            Y, 
            JSON_file, 
            Epochs
        )
        
        # * Compiles the model before training
        MLP.compile_model()

        # * Fits the model to the input and target data
        Model, Hist_data = MLP.fit_model()

        # * Save model to the folder
        ModelSaverDL.save_model(Model, Model_name)

        '''DataFrameCreatorDL.create_dataframe_history(
            Column_names, 
            Folder_save, 
            CSV_name, 
            Hist_data
        )'''

        # * Plot the training data for the model
        DataPlotterDL.plot_data_loss(Hist_data, Model_name, r'app\data')
        DataPlotterDL.plot_data_accuracy(Hist_data, Model_name, r'app\data')

    


