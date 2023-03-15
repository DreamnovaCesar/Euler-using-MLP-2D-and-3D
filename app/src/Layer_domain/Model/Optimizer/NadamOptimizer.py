from keras.optimizers import Nadam

from .Optimizer import Optimizer

class NadamOptimizer(Optimizer):
    def get_optimizer(learning_rate : float = 0.001):
        return Nadam(learning_rate = learning_rate)