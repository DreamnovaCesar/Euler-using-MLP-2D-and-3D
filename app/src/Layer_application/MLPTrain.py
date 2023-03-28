import numpy as np
from abc import ABC
from abc import abstractmethod

class MLPTrain(ABC):
    """
    An abstract base class for Multilayer perceptron training models.

    Parameters
    ----------
    X : ndarray
        A numpy array containing input data for training the MLP model.
    Y : ndarray
        A numpy array containing target data for training the MLP model.
    JSON_file : str
        The file path for saving the trained model as a JSON file.
    Model_name : str
        The name of the trained model.
    Epochs : int
        The number of epochs to train the MLP model.

    Returns
    -------
    None
    """
    
    @abstractmethod
    def train(
        self, 
        X : np.ndarray, 
        Y: np.ndarray, 
        JSON_file : str,
        Model_name :str, 
        Epochs: int
    ):
        
        """
        An abstract method to be implemented by subclasses for training an MLP model.

        Parameters
        ----------
        X : ndarray
            A numpy array containing input data for training the MLP model.
        Y : ndarray
            A numpy array containing target data for training the MLP model.
        JSON_file : str
            The file path for saving the trained model as a JSON file.
        Model_name : str
            The name of the trained model.
        Epochs : int
            The number of epochs to train the MLP model.

        Returns
        -------
        None
        """
        pass

