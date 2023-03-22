from keras.optimizers import Adagrad

from .Optimizer import Optimizer

class AdagradOptimizer(Optimizer):
    """A class for creating an Adagrad optimizer in Keras.
    
    Methods
    -------
    get_optimizer(learning_rate: float) -> optimizer:
        Abstract method that creates and returns a new optimizer instance.
    
    """

    def get_optimizer(learning_rate : float):
        """Creates an Adagrad optimizer in Keras.

        Parameters
        -------
            learning_rate (float): The learning rate for the optimizer.

        Returns
        -------
            Optimizer: A new optimizer instance.

        Examples
        -------
            >>> optimizer = AdagradOptimizer.get_optimizer(0.01)

        """
        return Adagrad(learning_rate = learning_rate)