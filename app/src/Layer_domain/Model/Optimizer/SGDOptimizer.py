from keras.optimizers import Adam

from .Optimizer import Optimizer

class SGDOptimizer(Optimizer):
    """A class for creating an SGD optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """
    
    def get_optimizer(learning_rate: float = 0.0000001):
        """Creates an SGD optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = SGDOptimizer.get_optimizer(0.01)

        """
        return Adam(learning_rate = learning_rate)