import numpy as np
from ..Layer_domain.Model.MLP import MLP
from ..Layer_domain.Json.JsonFileHander import JsonFileHandler
from ..Layer_domain.Model.ModelSaverDL import ModelSaverDL
from ..Layer_presentation.DataPlotterDL import DataPlotterDL
from ..Layer_domain.DataFrameCreatorDL import DataFrameCreatorDL
from ..Layer_domain.Model.ModelBuilderMLPI1 import ModelBuilderMLPI1

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

        self.MLP_training = MLP_training;

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
            Epochs,
            ModelBuilderMLPI1
        );
        
        # * Compiles the model before training
        MLP.compile_model();

        # * Read the model hyperparameters from the JSON file
        MLP_hp = JsonFileHandler.read_json_file(JSON_file);
        Opt = MLP_hp['optimizer'];
        AO = MLP_hp['activation_output'];
        Lr = MLP_hp['lr'];

        Model_name = 'Standard_{}_{}_{}_{}_{}'.format(Model_name, Opt, AO, Lr, Epochs)

        # * Fits the model to the input and target data
        Model, Hist_data, Folder_store_data = MLP.fit_model('Standard');

        # * Save model to the folder
        ModelSaverDL.save_model(Folder_store_data, Model, Model_name);

        # * Plot the training data for the model
        DataPlotterDL.plot_data_loss(Folder_store_data, Hist_data, Model_name);
        DataPlotterDL.plot_data_accuracy(Folder_store_data, Hist_data, Model_name);

    


