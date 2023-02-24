from keras.optimizers import Adam

from .Class_Optimizer import Optimizer

class SGDOptimizer(Optimizer):
    def get_optimizer(self, learning_rate: float = 0.0000001):
        return Adam(learning_rate = learning_rate)