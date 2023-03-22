from keras.optimizers import Ftrl

from .Optimizer import Optimizer

class FTRLOptimizer(Optimizer):
    """A class for creating an TRL optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """

    def get_optimizer(learning_rate : float):
        """Creates an TRL optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = FTRLOptimizer.get_optimizer(0.01)

        """
        return Ftrl(learning_rate = learning_rate)