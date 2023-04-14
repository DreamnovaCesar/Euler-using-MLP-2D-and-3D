from typing import List, Tuple, Dict
from typing import Union, Any

import os 
from .Model import Model
from .ModelBuilder import ModelBuilder
from ..Json.JsonFileHander import JsonFileHandler

from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ReduceLROnPlateau

from typing import Union

from ..Decorators.DisplayTraining import DisplayTraining

class MLP(Model):
    """
    Multi-layer perceptron (MLP) model for classification or regression problems.

    Parameters:
    -----------
    Input_shape : tuple[int, ...]
        The shape of the input data, excluding the batch size.
    Output_shape : tuple[int, ...]
        The shape of the output data, excluding the batch size.
    JSON_file : str
        The path to the JSON file containing the model architecture.
    Epochs : int
        The number of epochs to train the model.

    Methods:
    --------
    compile_model():
        Compiles the Keras model.
    fit_model():
        Trains the Keras model.
    predict_model(Model, Array) -> Union[None, Any]:
        Makes predictions with the Keras model.

    """

    def __init__(
        self, 
        Input_shape: Tuple[int, ...],
        Output_shape : Tuple[int, ...], 
        JSON_file : str,
        Epochs : int,
        ModelBuild : ModelBuilder
    ) -> None:
        
        """
        MLP class constructor.

        Parameters:
        -----------
        Input_shape : Tuple[int, ...]
            Tuple containing the shape of the input data.
        Output_shape : Tuple[int, ...]
            Tuple containing the shape of the output data.
        JSON_file : str
            File containing the model architecture in JSON format.
        Epochs : int
            Number of epochs for training the model.

        Returns:
        --------
        None
        """

        self.Input_shape = Input_shape;
        self.Output_shape = Output_shape;
        self.Epochs = Epochs;
        self.ModelBuild = ModelBuild;

        #print(self.Input_shape)
        #print(self.Output_shape)

        # * Read the model hyperparameters from the JSON file
        self.MLP_hp = JsonFileHandler.read_json_file(JSON_file);
        self.Opt = self.MLP_hp['optimizer'];
        self.Dense_1 = self.MLP_hp['dense_1'];
        self.Lr = self.MLP_hp['lr'];

        self.Model, self.Parameters = self.ModelBuild.build_model(
                                        self.Input_shape,
                                        JSON_file
                                    );

        print(self.Parameters);
    
    def compile_model(self) -> None:
        """
        Compile the neural network model.
        """
        
        self.Model.compile(
            optimizer = self.Parameters[0], 
            loss = self.Parameters[1], 
            metrics = [self.Parameters[2]]
        );

    @DisplayTraining.display
    def fit_model(self, TypeMLP) -> Tuple[Any, Dict[str, Any]]:
        """
        Train the neural network model.

        Parameters:
        -----------
        None

        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            A tuple containing the trained model and history data.
        """

        Folder_train = "Folder_Data_Models_{}_{}_{}_{}".format(TypeMLP, self.Opt, self.Lr, self.Dense_1)

        Folder_train_data = '{}/{}'.format(r'app\data', Folder_train)

        # *
        Exist_folder_train_data = os.path.isdir(Folder_train_data)

        # *
        if Exist_folder_train_data == False:
          Folder_path = os.path.join(r'app\data', Folder_train)
          os.mkdir(Folder_path)
          print(Folder_path)
        else:
          Folder_path = os.path.join(r'app\data', Folder_train)
          print(Folder_path)

        # * Save best model weights for each model.
        Best_model_name_weights = "Dataframe_Best_Model_Weights_{}_{}_{}.h5".format(self.Opt, self.Lr, self.Epochs)
        Best_model_folder_name_weights = os.path.join(Folder_path, Best_model_name_weights)

        # * Save dataframe Logger (data: Accuracy and loss) for each model.
        #CSV_logger_info = str(Class_problem_prefix) + str(Pretrained_model_name) + '_' + str(Enhancement_technique) + '.csv'
        CSV_logger_info = "Dataframe_Logger_{}_{}_{}.csv".format(self.Opt, self.Lr, self.Epochs)
        CSV_logger_info_folder = os.path.join(Folder_path, CSV_logger_info)

        # * Using ModelCheckpoint class.
        Model_checkpoint_callback = ModelCheckpoint(filepath = Best_model_folder_name_weights,
                                                    save_weights_only = True,                     
                                                    monitor = 'loss',
                                                    mode = 'max',
                                                    save_best_only = True )


        # * Using CSVLogger class to extract each epoch. 
        Log_CSV = CSVLogger(CSV_logger_info_folder, separator = ',', append = False)

        # * Save all callbacks to use them together
        Callbacks = [Model_checkpoint_callback, Log_CSV]

        Hist_data = self.Model.fit(
            self.Input_shape, 
            self.Output_shape, 
            batch_size = 8, 
            epochs = self.Epochs, 
            verbose = True,
            callbacks = Callbacks
        );

        return self.Model, Hist_data, Folder_path
    
    @staticmethod
    def predict_model(Model, Array) -> Union[None, Any]:
        """
        Generate predictions for input data using the trained model.

        Parameters:
        -----------
        Model : Any
            Trained neural network model.
        Array : np.ndarray
            Input data for which predictions are to be generated.

        Returns:
        --------
        Union[None, Any]
            Prediction generated by the neural network model.
        """

        return Model.predict(Array)
    