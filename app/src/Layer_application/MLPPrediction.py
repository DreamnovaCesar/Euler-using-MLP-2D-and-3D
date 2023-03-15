
from abc import ABC
from abc import abstractmethod

from ..Layer_domain.Model.MLP import MLP

class MLPPrediction(ABC):
    
    def __init__(self, 
                 MLP_prediction: MLP):
        
        self.MLP_prediction = MLP_prediction

    def prediction(self, Data):
        pass

