
from typing import List, Tuple

from Class_DataProcessor import DataProcessor
from ..Layer_domain.Model.Class_MLPModelBuilder import MLPModelBuilder
from ..Layer_domain.Model.Class_MLP import MLP

class TrainMLP(object):
    
    def __init__(self, MLP_model_builder : MLPModelBuilder, 
                        MLP_training : MLP):
        
        self.MLP_model_builder = MLP_model_builder
        self.MLP_training = MLP_training

        self.MLP_model_builder()
