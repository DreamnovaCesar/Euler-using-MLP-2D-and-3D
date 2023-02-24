from keras.optimizers import Adadelta

from .Class_Optimizer import Optimizer

class AdadeltaOptimizer(Optimizer):
    """A class for creating an Adadelta optimizer in Keras.

    """
    def get_optimizer(self, learning_rate: float = 0.0000001):
        """Creates an Adadelta optimizer in Keras.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate to use for the optimizer. Default is 0.0000001.

        Returns
        -------
        optimizer : Adadelta
            The created Adadelta optimizer.

        """
        return Adadelta(learning_rate = learning_rate)