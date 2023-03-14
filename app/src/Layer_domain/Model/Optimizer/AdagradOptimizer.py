from keras.optimizers import Adagrad

from .Optimizer import Optimizer

class AdagradOptimizer(Optimizer):
    def get_optimizer(self, learning_rate: float = 0.0000001):
        return Adagrad(learning_rate = learning_rate)