import os
import pandas as pd

from .Saver import Saver

class SaverCSV(Saver):
    """
    A class that saves a list of Data to a CSV file in a given folder path.

    Parameters
    ----------
    DataFrame : pandas.DataFrame
        The DataFrame to save the Data to.

    Methods
    -------
    save_file(Folder_path : str, Data : list[int])
        Appends the data to the DataFrame and saves it to a CSV file in the given folder path.
    """

    def __init__(self):
        """
        Initializes an empty DataFrame to store the Data.
        """
        self.DataFrame = pd.DataFrame()

    def save_file(self, Folder_path : str, Data : list[int]) -> pd.DataFrame:
        """
        Appends the Data to the DataFrame and saves it to a CSV file in the given folder path.

        Parameters
        ----------
        Folder_path : str
            The path of the folder where to save the CSV file.
        Data : list[int]
            The list of Data to append to the DataFrame.
        """

        # * Return the new dataframe with the new data
        self.DataFrame = self.DataFrame.append(pd.Series(Data), ignore_index = True)
            
        Dataframe_name = 'Dataframe_Data.csv'.format()
        Dataframe_folder = os.path.join(Folder_path, Dataframe_name)
        self.DataFrame.to_csv(Dataframe_folder)

        return self.DataFrame