from typing import List, Tuple
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

from .ModelBuilder import ModelBuilder

from ..Json.JsonFileHander import JsonFileHandler
from ...Layer_application.OptimizerOptions import OptimizerOptions

class ModelBuilderMLPI1(ModelBuilder):
    """
    A class for building a Multi-Layer Perceptron model using Keras.

    Methods
    -------
    build_model(Input_shape : Tuple[int, ...], JSON_file : str) -> Tuple[Sequential, List[str, str, str]]:
        Builds a multi-layer perceptron model using Keras.

    Examples
    --------
    >>> builder = ModelBuilderMLPI1()
    >>> input_shape = (32, 32, 3)
    >>> json_file_path = "/path/to/hyperparameters.json"
    >>> model, [optimizer, loss, metrics] = builder.build_model(input_shape, json_file_path)
    """

    @staticmethod
    def build_model(
        Input_shape : Tuple[int, ...],
        JSON_file : str
    ) -> Tuple[Sequential, List[str, str, str]]:
        """
        Builds a multi-layer perceptron model using Keras.

        Parameters
        ----------
        Input_shape : Tuple[int, ...]
            A tuple of integers representing the input shape of the model.
        JSON_file : str
            A string representing the path of the JSON file containing the model hyperparameters.

        Returns
        -------
        Tuple[Sequential, List[str, str, str]]
            A tuple of a Keras Sequential model object and a list of optimizer object, loss function object, and 
            metric object.

        """

        # * Read the model hyperparameters from the JSON file
        MLP_hp = JsonFileHandler.read_json_file(JSON_file);

        # * Extract the hyperparameters from the dictionary
        dense_1 = MLP_hp['dense_1'];
        output = MLP_hp['output'];
        activation_1 = MLP_hp['activation_1'];
        activation_output = MLP_hp['activation_output'];
        Opt = MLP_hp['optimizer'];
        Lr = MLP_hp['lr'];
        Loss = MLP_hp['loss'];
        Metrics = MLP_hp['metrics'];

        # * Choose the optimizer based on the optimizer option and learning rate
        Optimizer = OptimizerOptions.choose_optimizer(Opt, Lr);
        
        # * Define the Keras Sequential model
        Model = Sequential()
        Model.add(Input(shape = Input_shape.shape[1],))
        Model.add(Dense(dense_1, activation = activation_1))
        Model.add(Dense(output, activation = activation_output))

        # * Return the model, optimizer, loss function, and metric objects
        return Model, [Optimizer, Loss, Metrics]
    