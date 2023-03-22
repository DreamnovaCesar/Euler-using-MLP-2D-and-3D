
import os
from .ModelSaver import ModelSaver

class ModelSaverML(ModelSaver):

    def save_model(
        self, 
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
        Model_name_joblib = '{}.joblib'.format(Model_name)
        Model_folder = os.path.join(r'app\data', Model_name_joblib)
        Model.save(Model_folder)

        # * Prints that the model has been saved
        print("Saving model...")
        print('\n')

