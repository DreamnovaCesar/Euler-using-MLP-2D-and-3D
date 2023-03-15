from keras.optimizers import Ftrl

from .Optimizer import Optimizer

class FtrlOptimizer(Optimizer):
    
    def get_optimizer(learning_rate : float):
        return Ftrl(learning_rate = learning_rate)