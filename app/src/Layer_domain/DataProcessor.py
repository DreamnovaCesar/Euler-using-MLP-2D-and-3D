import numpy as np 

from .CSVDataLoader import CSVReader

class DataProcessor:
    def __init__(self, Data_reader : CSVReader, 
                 initial_x : int = 1, 
                 final_x : int = 257):
        
        self.Data_reader = Data_reader
        self.initial_x = initial_x
        self.final_x = final_x

    def process_data(self, source):
        # read the data from the source using the data reader
        dataframe = self.Data_reader.load_data(source)

        # extract the input data and labels from the dataframe
        X = dataframe.iloc[:, self.initial_x:self.final_x].values
        Y = dataframe.iloc[:, -1].values

        # reshape the labels array to have the same number of dimensions as the input data
        Y = np.expand_dims(Y, axis = 1)

        return X, Y