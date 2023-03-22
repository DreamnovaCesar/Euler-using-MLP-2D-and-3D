
import os
from .ModelSaver import ModelSaver

class ModelSaverDL(ModelSaver):

    def save_model(
        self, 
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
        Model_name_h5 = '{}.h5'.format(Model_name)
        Model_folder = os.path.join(r'app\data', Model_name_h5)
        Model.save(Model_folder)

        # * Prints that the model has been saved
        print("Saving model...")
        print('\n')

