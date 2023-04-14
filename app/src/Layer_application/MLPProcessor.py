
from ..Layer_domain.DataProcessor import DataProcessor
from ..Layer_domain.Model.MLP import MLP
from ..Layer_domain.Model.ModelBuilderMLPV2 import ModelBuilderMLPV2
from ..Layer_domain.Json.JsonFileHander import JsonFileHandler

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

    def __init__(
        self, 
        Data_processor : DataProcessor,
        MLP_training : MLP
    ):

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

        Processor = self.Data_processor(DataLoaderCSV);

        X, Y = Processor.process_data(CSV_file);

        # * Instantiates an MLP object to train
        MLP = self.MLP_training(
            X, 
            Y, 
            JSON_file, 
            Epochs,
            ModelBuilderMLPV2
        );
        
        # * Read the model hyperparameters from the JSON file
        MLP_hp = JsonFileHandler.read_json_file(JSON_file);
        Opt = MLP_hp['optimizer'];
        Lr = MLP_hp['lr'];

        Model_name = '{}_{}_{}_{}'.format(Model_name, Opt, Lr, Epochs)

        # * Compiles the model before training
        MLP.compile_model();

        # * Fits the model to the input and target data
        Model, Hist_data, Folder_store_data = MLP.fit_model('Processor');

        # * Save model to the folder
        ModelSaverDL.save_model(Folder_store_data, Model, Model_name);

        # * Plot the training data for the model
        DataPlotterDL.plot_data_loss(Folder_store_data, Hist_data, Model_name);
        #DataPlotterDL.plot_data_accuracy(Hist_data, Model_name, r'app\data');

