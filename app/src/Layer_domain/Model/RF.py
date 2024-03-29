import numpy as np
from typing import List, Tuple
from typing import Union, Any

from .ModelML import ModelML
from ..Json.JsonFileHander import JsonFileHandler

from sklearn.ensemble import RandomForestClassifier

class RF(ModelML):
    """
    Random Forest (RF) model for classification.

    Parameters:
    -----------
    Input_shape : tuple[int, ...]
        The shape of the input data, excluding the batch size.
    Output_shape : tuple[int, ...]
        The shape of the output data, excluding the batch size.
    JSON_file : str
        The path to the JSON file containing the model architecture.

    Methods:
    --------
    fit_model():
        Trains the RF model.
    predict_model(Model, Array) -> Union[None, Any]:
        Makes predictions with the RF model.

    """

    def __init__(
        self, 
        Input_shape: Tuple[int, ...],
        Output_shape : Tuple[int, ...], 
        JSON_file : str,
    ):
        """
        RF class constructor.

        Parameters:
        -----------
        Input_shape : Tuple[int, ...]
            Tuple containing the shape of the input data.
        Output_shape : Tuple[int, ...]
            Tuple containing the shape of the output data.
        JSON_file : str
            File containing the model architecture in JSON format.

        Returns:
        --------
        None
        """

        self.Input_shape = Input_shape;
        self.Output_shape = Output_shape;
        
        # * Read the model hyperparameters from the JSON file
        MLP_hp = JsonFileHandler.read_json_file(JSON_file);

        # * Extract the hyperparameters from the dictionary
        Criterion = MLP_hp['criterion'];
        N_estimators = MLP_hp['n_estimators'];
        Random_state = MLP_hp['random_state'];
        N_jobs = MLP_hp['n_jobs'];
        
        self.Model = RandomForestClassifier(
            criterion = Criterion,
            n_estimators = N_estimators,
            random_state = Random_state,
            n_jobs = N_jobs
        ) ;

    def fit_model(self):
        """
        Train the random forest.

        Parameters:
        -----------
        None

        Returns:
        --------
        Tuple[Any, Dict[str, Any]]
            A tuple containing the trained model and history data.
        """
        return self.Model.fit(
            self.Input_shape, 
            self.Output_shape
        );
    
    @staticmethod
    def predict_model(Model, Array : np.ndarray) -> Union[None, Any]:
        """
        Generate predictions for input data using the trained model.

        Parameters:
        -----------
        Array : np.ndarray
            Input data for which predictions are to be generated.

        Returns:
        --------
        Union[None, Any]
            Prediction generated by the neural network model.
        """

        return Model.predict(Array)
    
