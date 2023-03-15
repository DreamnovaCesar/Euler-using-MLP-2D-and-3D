
from typing import List, Tuple
import os 

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

from ..Layer_domain.Model.ModelBuilderMLPV2 import ModelBuilderMLPV2

from ..Layer_domain.DataLoaderCSV import DataLoaderCSV
from ..Layer_presentation.DataPlotterDL import DataPlotterDL

from .MLPTrain import MLPTrain

class MLPProcessor(MLPTrain):
    
    def __init__(self, 
                 Data_processor : DataProcessor,
                 MLP_training : MLP):
        
        self.Data_processor = Data_processor
        self.MLP_training = MLP_training

    def train(self, JSON_file, CSV_file : str, Model_name, epochs = 10000, Lr = 0.1):

        # * Prints that training has completed
        print('\n')
        print("Model trained")
        print('\n')

        Data = self.Data_processor(DataLoaderCSV)
        X, Y = Data.process_data(CSV_file)
        MLP = self.MLP_training(X, 
                                Y, 
                                JsonFileHandler, 
                                JSON_file, 
                                AdamOptimizer,
                                ModelBuilderMLPV2,
                                epochs,
                                Lr)
        
        MLP.compile_model()
        Model, Hist_data = MLP.fit_model()

        # * Prints that training has begun
        print('\n')
        print("Training...")
        print('\n')

        # * Save the trained model as an h5 file
        Model_name_save = '{}.h5'.format(Model_name)
        Model_folder_save = os.path.join(r'app\data', Model_name_save)

        Model.save(Model_folder_save)

        DataPlotterDL.plot_data_loss(Hist_data, Model_name, r'app\data')
        DataPlotterDL.plot_data_accuracy(Hist_data, Model_name, r'app\data')

