from typing import List, Tuple
from typing import Union, Any

from .Class_Model import Model

from sklearn.ensemble import RandomForestClassifier

class RF(Model):
    def __init__(self, criterion: str, n_estimators: int, n_jobs: int):
        # Create a Random Forest model
        self.model = RandomForestClassifier( criterion = criterion,
                                            n_estimators = n_estimators,
                                            random_state = 2,
                                            n_jobs = n_jobs) 

    def fit(self, x, y, verbose):
        return self.model.fit(x, y, verbose=verbose)
    
    def predict(self, data) -> Union[None, Any]:
        pass
    
