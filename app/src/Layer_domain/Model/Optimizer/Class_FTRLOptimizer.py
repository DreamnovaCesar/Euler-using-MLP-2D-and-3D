from keras.optimizers import Ftrl

from .Class_Optimizer import Optimizer

class FtrlOptimizer(Optimizer):
    def get_optimizer(self, learning_rate: float = 0.0000001):
        return Ftrl(learning_rate = learning_rate)