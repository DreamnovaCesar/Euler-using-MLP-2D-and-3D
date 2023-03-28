from abc import ABC
from abc import abstractmethod

class ModelSaver(ABC):

    @staticmethod
    @abstractmethod
    def save_model(
        Model : object, 
        Model_name : str
    ):
        
        """
        Abstract method to save a trained model to disk.

        Parameters:
        -----------
        Model : object
            Trained neural network model.
        Model_name : str
            Name of the file to save the model.

        Returns:
        --------
        None
        """
        pass