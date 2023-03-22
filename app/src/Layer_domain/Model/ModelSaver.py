from abc import ABC
from abc import abstractmethod

from ..Decorators.DisplayModelSave import DisplayModelSave
class ModelSaver(ABC):

    @staticmethod
    @abstractmethod
    @DisplayModelSave.display
    def save_model(
        self, 
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