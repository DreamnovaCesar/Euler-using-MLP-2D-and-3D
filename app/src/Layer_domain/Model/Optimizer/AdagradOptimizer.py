from keras.optimizers import Adagrad

from .Optimizer import Optimizer

class AdagradOptimizer(Optimizer):
    def get_optimizer(learning_rate : float):
        return Adagrad(learning_rate = learning_rate)