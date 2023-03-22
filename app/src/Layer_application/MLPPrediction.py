
import numpy as np

from abc import ABC
from abc import abstractmethod

from ..Layer_domain.Model.MLP import MLP

class MLPPrediction(ABC):
    """
    An abstract base class for MLP prediction models.

    Attributes
    ----------
    None

    Methods
    -------
    prediction()
        An abstract method to be implemented by subclasses for making predictions based on input data.
    """

    def __init__(
        self, 
        MLP_prediction: MLP
    ):
        
        """
        Initializes an instance of the MLPPrediction class.

        Parameters
        ----------
        MLP_prediction : MLP
            An MLP object used for making predictions.
        """

        self.MLP_prediction = MLP_prediction

    @property
    @abstractmethod
    def prediction(
        self, 
        Data : np.ndarray
    ):
        
        """
        An abstract method to be implemented by subclasses for making predictions based on input data.

        Parameters
        ----------
        Data : ndarray
            A numpy array containing input data for making predictions.

        Returns
        -------
        None
        """
        pass

