
from .Class_ModelSaver import ModelSaver

class ModelSaverML(ModelSaver):
    def __init__(self, file_saver):
        self.file_saver = file_saver

    def save_model(self, model, model_name):
        # Save the trained model as an h5 file
        Model_name_save = '{}.joblib'.format(model_name)
        Model_folder_save = self.file_saver.save_file(Model_name_save)
        model.save(Model_folder_save)

        # Prints that the model has been saved
        print("Saving model...")
        print('\n')

