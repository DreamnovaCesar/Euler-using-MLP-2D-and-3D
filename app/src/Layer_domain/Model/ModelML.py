from abc import ABC
from abc import abstractmethod

from typing import Union, Any

class ModelML(ABC):
    """An abstract base class for defining a machine learning model.

    This class defines three abstract methods that need to be implemented by any concrete model class:
    `fit()`, and `predict()`.

    """

    @abstractmethod
    def fit_model(self, X, Y, verbose):
        """Fit the model to the training data.

        Parameters
        ----------
        X : array-like
            The training data.
        Y : array-like
            The target data.
        verbose : bool
            Whether or not to print verbose output during training.

        Returns
        -------
        None

        """
        pass
    

    @abstractmethod
    def predict_model(self, Array) -> Union[None, Any]:
        """Use the trained model to make predictions.

        Parameters
        ----------
        Array : array-like
            The data to make predictions on.

        Returns
        -------
        Union[None, Any]
            The predictions made by the model, or None if the model hasn't been trained yet.

        """
        pass