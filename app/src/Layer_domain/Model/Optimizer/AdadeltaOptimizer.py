from keras.optimizers import Adadelta

from .Optimizer import Optimizer

class AdadeltaOptimizer(Optimizer):
    """A class for creating an Adadelta optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """
    
    def get_optimizer(learning_rate : float):
        """Creates an Adadelta optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = AdadeltaOptimizer.get_optimizer(0.01)

        """
        return Adadelta(learning_rate = learning_rate)