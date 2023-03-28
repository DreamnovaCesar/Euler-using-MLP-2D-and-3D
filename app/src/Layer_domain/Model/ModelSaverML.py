
import os
import joblib
from .ModelSaver import ModelSaver

from ..Decorators.DisplayModelSave import DisplayModelSave

class ModelSaverML(ModelSaver):

    @staticmethod
    @DisplayModelSave.display
    def save_model(
        Model : object, 
        Model_name : str
    ) -> None:
        """
        Save a machine learning model using joblib library.

        Parameters:
        -----------
        Model : object
            The trained machine learning model to be saved.
        Model_name : str
            The name to be given to the saved model.

        Returns:
        --------
        None
        """

        # * Save the trained model as an h5 file
        Model_name_joblib = '{}_RF.joblib'.format(Model_name);
        Model_folder = os.path.join(r'app\data', Model_name_joblib);
        joblib.dump(Model, Model_folder);


