from keras.optimizers import Nadam

from .Optimizer import Optimizer

class NadamOptimizer(Optimizer):
    def get_optimizer(self, learning_rate: float = 0.0000001):
        return Nadam(learning_rate = learning_rate)