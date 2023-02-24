from keras.optimizers import RMSprop

from .Class_Optimizer import Optimizer

class RMSpropOptimizer(Optimizer):
    def get_optimizer(self, learning_rate: float = 0.0000001):
        return RMSprop(learning_rate = learning_rate)