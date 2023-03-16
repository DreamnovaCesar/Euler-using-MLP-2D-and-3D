import os
import pandas as pd

from .Saver import Saver

class SaverCSV(Saver):
    def __init__(self):
        self.DataFrame = pd.DataFrame()

    def save_file(self, Folder_path : str, Octovoxels : list[int]):
            
        # * Return the new dataframe with the new data
        self.DataFrame = self.DataFrame.append(pd.Series(Octovoxels), ignore_index = True)
            
        Dataframe_name = 'Dataframe_Data.csv'.format()
        Dataframe_folder = os.path.join(Folder_path, Dataframe_name)
        self.DataFrame.to_csv(Dataframe_folder)