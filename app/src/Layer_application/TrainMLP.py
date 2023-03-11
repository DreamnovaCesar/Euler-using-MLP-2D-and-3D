
from typing import List, Tuple

from ..Layer_domain.DataProcessor import DataProcessor
from ..Layer_domain.Model.ModelBuilderMLP import MLPModelBuilder
from ..Layer_domain.Model.MLP import MLP

class TrainMLP(object):
    
    def __init__(self, 
                 Data_processor : DataProcessor,
                 MLP_model_builder : MLPModelBuilder, 
                 MLP_training : MLP):
        
        self.MLP_model_builder = MLP_model_builder
        self.MLP_training = MLP_training

        self.MLP_model_builder()
