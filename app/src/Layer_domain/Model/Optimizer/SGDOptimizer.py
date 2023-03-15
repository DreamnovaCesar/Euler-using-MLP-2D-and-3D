from keras.optimizers import Adam

from .Optimizer import Optimizer

class SGDOptimizer(Optimizer):
    
    def get_optimizer(learning_rate: float = 0.0000001):
        return Adam(learning_rate = learning_rate)