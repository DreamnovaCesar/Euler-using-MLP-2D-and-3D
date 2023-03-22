from keras.optimizers import Adam

from .Optimizer import Optimizer

class AdamOptimizer(Optimizer):
    """A class for creating an Adam optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """

    def get_optimizer(learning_rate : float):
        """Creates an Adam optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = AdamOptimizer.get_optimizer(0.01)

        """
        return Adam(learning_rate = learning_rate)