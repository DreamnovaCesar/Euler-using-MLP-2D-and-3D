from typing import List, Tuple
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

from .ModelBuilder import ModelBuilder

class ModelBuilderMLP(ModelBuilder):
    """A class for building a Multi-Layer Perceptron model using Keras.
    """

    @staticmethod
    def build_model(input_shape : Tuple[int, ...], 
                    dense_1 : int, 
                    output : int, 
                    activation_1 : str, 
                    activation_output : str) -> Sequential:

        """Builds a Multi-Layer Perceptron model using Keras.

        Parameters
        ----------
        input_shape : Tuple[int, ...]
            The shape of the input data.
        dense_1 : int
            The number of units in the first dense layer.
        output : int
            The number of units in the output layer.
        activation_1 : str
            The activation function to use in the first dense layer.
        activation_output : str
            The activation function to use in the output layer.

        Returns
        -------
        model : Sequential
            The built MLP model.

        """

        model = Sequential()
        model.add(Input(shape = input_shape))
        model.add(Dense(dense_1, activation = activation_1))
        model.add(Dense(output, activation = activation_output))

        return model
    