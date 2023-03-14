
from typing import List, Tuple

from ..Layer_domain.DataProcessor import DataProcessor
from ..Layer_domain.Model.MLP import MLP

from ..Layer_domain.Json.JsonFileHander import JsonFileHandler
from ..Layer_domain.Model.Optimizer.AdadeltaOptimizer import AdadeltaOptimizer
from ..Layer_domain.Model.Optimizer.AdagradOptimizer import AdagradOptimizer
from ..Layer_domain.Model.Optimizer.AdamaxOptimizer import AdamaxOptimizer
from ..Layer_domain.Model.Optimizer.AdamOptimizer import AdamOptimizer
from ..Layer_domain.Model.Optimizer.FTRLOptimizer import FtrlOptimizer
from ..Layer_domain.Model.Optimizer.NadamOptimizer import NadamOptimizer
from ..Layer_domain.Model.Optimizer.RMSpropOptimizer import RMSpropOptimizer
from ..Layer_domain.Model.Optimizer.SGDOptimizer import SGDOptimizer

from ..Layer_domain.DataLoaderCSV import DataLoaderCSV

from .MLPTrain import MLPTrain

class MLPProcessing(MLPTrain):
    
    def __init__(self, 
                 Data_processor : DataProcessor,
                 MLP_training : MLP):
        
        self.Data_processor = Data_processor
        self.MLP_training = MLP_training

    def train(self, JSON_file, Path : str):

        # * Prints that training has completed
        print('\n')
        print("Model trained")
        print('\n')

        Data = self.Data_processor(DataLoaderCSV)
        X, Y = Data.process_data(Path)
        MLP = self.MLP_training(X, Y, JsonFileHandler, JSON_file, AdamaxOptimizer)
        MLP.fit_model()

        # * Prints that training has begun
        print('\n')
        print("Training...")
        print('\n')


