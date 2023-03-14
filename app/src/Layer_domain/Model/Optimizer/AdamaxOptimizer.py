from keras.optimizers import Adamax

from .Optimizer import Optimizer

class AdamaxOptimizer(Optimizer):
    def get_optimizer(self, learning_rate: float = 0.0000001):
        return Adamax(learning_rate = learning_rate)