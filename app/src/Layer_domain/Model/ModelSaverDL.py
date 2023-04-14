
import os
from .ModelSaver import ModelSaver

from ..Decorators.DisplayModelSave import DisplayModelSave

class ModelSaverDL(ModelSaver):

    @staticmethod
    @DisplayModelSave.display
    def save_model(
        Folder_path : str,
        Model : object, 
        Model_name : str
    ) -> None:
        
        """
        Save a neural network model in h5 format.

        Parameters:
        -----------
        Model : object
            The trained neural network model to be saved.
        Model_name : str
            The name to be given to the saved model.

        Returns:
        --------
        None
        """

        # * Save the trained model as an h5 file
        Model_name_h5 = '{}_MLP.h5'.format(Model_name);
        Model_folder = os.path.join(Folder_path, Model_name_h5);
        Model.save(Model_folder);

