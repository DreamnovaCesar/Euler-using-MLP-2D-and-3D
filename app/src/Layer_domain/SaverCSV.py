import os
import pandas as pd

from .Saver import Saver

class SaverCSV(Saver):
    """
    A class that saves a list of Octovoxels to a CSV file in a given folder path.

    Parameters
    ----------
    DataFrame : pandas.DataFrame
        The DataFrame to save the Octovoxels to.

    Methods
    -------
    save_file(Folder_path : str, Octovoxels : list[int])
        Appends the Octovoxels to the DataFrame and saves it to a CSV file in the given folder path.
    """

    def __init__(self):
        """
        Initializes an empty DataFrame to store the Octovoxels.
        """
        self.DataFrame = pd.DataFrame()

    def save_file(self, Folder_path : str, Octovoxels : list[int]):
        """
        Appends the Octovoxels to the DataFrame and saves it to a CSV file in the given folder path.

        Parameters
        ----------
        Folder_path : str
            The path of the folder where to save the CSV file.
        Octovoxels : list[int]
            The list of Octovoxels to append to the DataFrame.
        """

        # * Return the new dataframe with the new data
        self.DataFrame = self.DataFrame.append(pd.Series(Octovoxels), ignore_index = True)
            
        Dataframe_name = 'Dataframe_Data.csv'.format()
        Dataframe_folder = os.path.join(Folder_path, Dataframe_name)
        self.DataFrame.to_csv(Dataframe_folder)