from keras.optimizers import Adamax

from .Optimizer import Optimizer

class AdamaxOptimizer(Optimizer):
    """A class for creating an Adamax optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """

    def get_optimizer(learning_rate : float):
        """Creates an Adamax optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = AdamaxOptimizer.get_optimizer(0.01)

        """
        return Adamax(learning_rate = learning_rate)