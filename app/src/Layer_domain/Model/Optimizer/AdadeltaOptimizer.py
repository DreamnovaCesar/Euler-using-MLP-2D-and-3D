from keras.optimizers import Adadelta

from .Optimizer import Optimizer

class AdadeltaOptimizer(Optimizer):
    """A class for creating an Adadelta optimizer in Keras.

    """
    def get_optimizer(learning_rate : float):
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