from keras.optimizers import Nadam

from .Optimizer import Optimizer

class NadamOptimizer(Optimizer):
    """A class for creating an Nadam optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """

    def get_optimizer(learning_rate : float = 0.001):
        """Creates an Nadam optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = NadamOptimizer.get_optimizer(0.01)

        """
        return Nadam(learning_rate = learning_rate)