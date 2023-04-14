
from ..Layer_domain.Model.Optimizer.AdamOptimizer import AdamOptimizer
from ..Layer_domain.Model.Optimizer.AdagradOptimizer import AdagradOptimizer
from ..Layer_domain.Model.Optimizer.AdamaxOptimizer import AdamaxOptimizer
from ..Layer_domain.Model.Optimizer.AdadeltaOptimizer import AdadeltaOptimizer
from ..Layer_domain.Model.Optimizer.FTRLOptimizer import FTRLOptimizer
from ..Layer_domain.Model.Optimizer.NadamOptimizer import NadamOptimizer
from ..Layer_domain.Model.Optimizer.RMSpropOptimizer import RMSpropOptimizer
from ..Layer_domain.Model.Optimizer.SGDOptimizer import SGDOptimizer

from keras.optimizers import Adam

class OptimizerOptions(object):
    """
    Creates an optimizer object that can be used to optimize a model's parameters.

    Parameters
    ----------
    Name : str
        Name of the optimizer. Must be one of "ADAM", "NADAM", "ADAMAX", "ADAGRAD",
        "ADADELTA", "SGD", "RMSPROP", or "FTRL".
    lr : float
        Learning rate of the optimizer.

    Raises
    ------
    ValueError
        If the Name parameter is not a valid optimizer name.

    Examples
    --------
    Create an optimizer with the ADAM algorithm and learning rate 0.001:

    >>> opt = Optimizer("ADAM", 0.001)
    >>> opt.create_optimizer()
    <tensorflow.python.keras.optimizer_v2.adam.Adam object at 0x7f85d8ed12e8>
    """
    
    def __init__(
        self, 
        Name: str, 
        lr: float
    ):

        self.Name = Name
        self.lr = lr


        self.Name = self.Name.upper()

    def choose_optimizer(self):
        """
        Returns an optimizer object based on the Name parameter passed to the constructor.

        Returns
        -------
        optimizer : tensorflow.keras.optimizers.Optimizer
            An optimizer object that can be used to optimize a model's parameters.
        """

        if self.Name == "ADAM":
            print(self.Name);
            return AdamOptimizer.get_optimizer(self.lr)
        
        elif self.Name == "NADAM":
            print(self.Name);
            return NadamOptimizer.get_optimizer(self.lr)
        
        elif self.Name == "ADAMAX":
            print(self.Name);
            return AdamaxOptimizer.get_optimizer(self.lr)
        
        elif self.Name == "ADAGRAD":
            print(self.Name);
            return AdagradOptimizer.get_optimizer(self.lr)
        
        elif self.Name == "ADADELTA":
            print(self.Name);
            return AdadeltaOptimizer.get_optimizer(self.lr)
        
        elif self.Name == "SGD":
            print(self.Name);
            return SGDOptimizer.get_optimizer(self.lr)
        
        elif self.Name == "RMSPROP":
            print(self.Name);
            return RMSpropOptimizer.get_optimizer(self.lr)
        
        elif self.Name == "FTRL":
            print(self.Name);
            return FTRLOptimizer.get_optimizer(self.lr)
        
        else:
            raise ValueError("Invalid optimizer name.")