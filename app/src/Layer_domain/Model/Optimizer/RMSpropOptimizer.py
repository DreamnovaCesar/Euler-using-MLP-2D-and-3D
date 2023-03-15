from keras.optimizers import RMSprop

from .Optimizer import Optimizer

class RMSpropOptimizer(Optimizer):
    
    def get_optimizer(learning_rate: float = 0.0000001):
        return RMSprop(learning_rate = learning_rate)