
from ..Layer_domain.DataProcessor import DataProcessor
from ..Layer_domain.Model.MLP import MLP

from ..Layer_domain.DataLoaderCSV import DataLoaderCSV
from ..Layer_domain.Model.ModelSaverDL import ModelSaverDL
from ..Layer_presentation.DataPlotterDL import DataPlotterDL

from .MLPTrain import MLPTrain

class MLPProcessor(MLPTrain):
    """
    MLPProcessor class to train a multilayer perceptron (MLP) model.

    Parameters
    ----------
    Data_processor : DataProcessor
        The data processor object.
    MLP_training : MLP
        The MLP object for training.

    Methods
    -------
    train(CSV_file, JSON_file, Model_name, Epochs=10000)
        Trains the MLP model using the standard training algorithm.

    """

    def __init__(self, 
                 Data_processor : DataProcessor,
                 MLP_training : MLP):
        """
        Initializes the MLPProcessor class.

        Parameters
        ----------
        Data_processor : DataProcessor
            The data processor object.
        MLP_training : MLP
            The MLP object for training.

        Returns
        -------
        None

        """

        self.Data_processor = Data_processor
        self.MLP_training = MLP_training

    def train(
        self, 
        CSV_file : str, 
        JSON_file : str, 
        Model_name : str,
        Epochs : int = 10000, 
    ):
        """
        Trains the multilayer perceptron (MLP) model using the standard training algorithm.

        Parameters
        ----------
        CSV_file : str
            The file path to the CSV file containing the input and target data.
        JSON_file : str
            The file path to the JSON file containing the model's hyperparameters.
        Model_name : str
            The name of the model to be saved.
        Epochs : int, optional
            The number of training epochs. Default is 10000.

        Returns
        -------
        None

        """

        Processor = self.Data_processor(DataLoaderCSV)

        X, Y = Processor.process_data(CSV_file)

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

        # * Plot the training data for the model
        DataPlotterDL.plot_data_loss(Hist_data, Model_name, r'app\data')
        #DataPlotterDL.plot_data_accuracy(Hist_data, Model_name, r'app\data')

