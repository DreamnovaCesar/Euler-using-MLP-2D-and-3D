import os
import pandas as pd
from typing import List

from .DataFrameCreator import DataFrameCreator

class DataFrameCreatorDL(DataFrameCreator):
    """
    Create a dataframe from the trained model's history data and save it to a CSV file.

    Parameters
    ----------
    Column_names : List[str]
        List of column names for the dataframe.
    Folder_save : str 
        Path of the folder where the dataframe will be saved.
    CSV_name : str
        Name of the CSV file to be created.
    Hist_data : object
        History data from the trained model to be added to the dataframe.

    Notes
    -----
    The CSV file will be saved with the name `Dataframe_<CSV_name>_Loss_And_Accuracy.csv`
    in the specified `Folder_save` directory. The first column of the dataframe will contain
    the loss values, and the second column will contain the accuracy values.
    """
    
    def create_dataframe_history(
        Column_names: List[str], 
        Folder_save: str, 
        CSV_name: str, 
        Hist_data: object
    ) -> None:
        
        """
        Method to create dataframe from the trained model

        Parameters
        ----------
        Column_names : str
            Column names for the dataframe
        Folder_save : str 
            Folder to save the dataframe
        CSV_name : str
            Name of the CSV file
        Hist_data : object
            History data from the trained model to be added to the dataframe
        """
        
        # * Create an empty dataframe with column names for the dataframe
        Dataframe_created = pd.DataFrame(columns = Column_names)

        # * Extract loss and accuracy data from the history
        Accuracy = Hist_data.history["accuracy"]
        Loss = Hist_data.history["loss"]

        # * Combine loss and accuracy data
        History_data = zip(Loss, Accuracy)

        # * Add the combined data to the dataframe
        for _, (Loss_, Accuracy_) in enumerate(History_data):
            Dataframe_created.loc[len(Dataframe_created.index)] = [Loss_, Accuracy_]

        # * Save the dataframe to a CSV file
        Dataframe_name = "Dataframe_{}_Loss_And_Accuracy.csv".format(CSV_name)
        Dataframe_folder = os.path.join(Folder_save, Dataframe_name)
        Dataframe_created.to_csv(Dataframe_folder)