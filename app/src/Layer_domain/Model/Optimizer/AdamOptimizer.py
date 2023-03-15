from keras.optimizers import Adam

from .Optimizer import Optimizer

class AdamOptimizer(Optimizer):
    def get_optimizer(learning_rate : float):
        return Adam(learning_rate = learning_rate)