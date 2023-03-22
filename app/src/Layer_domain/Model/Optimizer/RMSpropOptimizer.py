from keras.optimizers import RMSprop

from .Optimizer import Optimizer

class RMSpropOptimizer(Optimizer):
    """A class for creating an RMS optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """

    def get_optimizer(learning_rate: float = 0.0000001):
        """Creates an RMS optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = RMSOptimizer.get_optimizer(0.01)

        """
        return RMSprop(learning_rate = learning_rate)