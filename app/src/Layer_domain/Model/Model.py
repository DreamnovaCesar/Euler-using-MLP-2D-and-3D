from abc import ABC
from abc import abstractmethod

from typing import Union, Any

class Model(ABC):
    """An abstract base class for defining a machine learning model.

    This class defines three abstract methods that need to be implemented by any concrete model class:
    `compile()`, `fit()`, and `predict()`.

    """

    @abstractmethod
    def compile_model(self, optimizer, loss, metrics):
        """Compile the model with an optimizer, loss function, and metrics.

        Parameters
        ----------
        optimizer : str
            The optimizer to use.
        loss : str
            The loss function to use.
        metrics : list of str
            The metrics to use.

        Returns
        -------
        None

        """
        pass

    @abstractmethod
    def fit_model(self, x, y, epochs, verbose):
        """Fit the model to the training data.

        Parameters
        ----------
        x : array-like
            The training data.
        y : array-like
            The target data.
        epochs : int
            The number of epochs to train the model.
        verbose : bool
            Whether or not to print verbose output during training.

        Returns
        -------
        None

        """
        pass
    

    @abstractmethod
    def predict_model(self, data) -> Union[None, Any]:
        """Use the trained model to make predictions.

        Parameters
        ----------
        data : array-like
            The data to make predictions on.

        Returns
        -------
        Union[None, Any]
            The predictions made by the model, or None if the model hasn't been trained yet.

        """
        pass