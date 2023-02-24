import numpy as np
from abc import ABC, abstractmethod

class Model(ABC):
    @abstractmethod
    def predict(self, arrays: np.ndarray) -> np.ndarray:
        pass

class KerasModel(Model):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)
    
    def predict(self, arrays: np.ndarray) -> np.ndarray:
        return self.model.predict(arrays)
    
class ScikitLearnModel(Model):
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
    
    def predict(self, arrays: np.ndarray) -> np.ndarray:
        return self.model.predict(arrays)
    
class Predictor:
    def __init__(self, model: Model):
        self.model = model
        
    def predict(self, arrays: np.ndarray) -> np.ndarray:
        return self.model.predict(arrays)
    
class ResultPrinter:
    def print_results(self, arrays: np.ndarray, results: np.ndarray) -> None:
        for array, result in zip(arrays, results):
            print('{} -------------- {}'.format(array, result))
            print('The result is: {}'.format(result))
            print('The true value is: {}'.format(result))
            print('\n')
