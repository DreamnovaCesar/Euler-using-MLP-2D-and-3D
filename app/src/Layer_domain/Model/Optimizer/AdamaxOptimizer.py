from keras.optimizers import Adamax

from .Optimizer import Optimizer

class AdamaxOptimizer(Optimizer):
    def get_optimizer(learning_rate : float):
        return Adamax(learning_rate = learning_rate)