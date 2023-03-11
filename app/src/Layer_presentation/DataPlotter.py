from abc import ABC
from abc import abstractmethod

class DataPlotter(ABC):
    """
    An abstract class that defines an interface for plotting data.

    Methods:
    --------
    plot_data_loss(Hist_data: object) -> None:
        Abstract method that must be implemented by subclasses to plot data loss.

    plot_data_accuracy(Hist_data: object) -> None:
        Abstract method that must be implemented by subclasses to plot data accuracy.

    """

    @staticmethod
    @abstractmethod
    def plot_data_loss(self, Hist_data: object) -> None:
        """
        Abstract method that must be implemented by subclasses to plot data loss.

        Parameters:
        -----------
        Hist_data: object
            The object containing the loss data to be plotted.

        Returns:
        --------
        None
        """
        pass
    
    @staticmethod
    @abstractmethod
    def plot_data_accuracy(self, Hist_data: object) -> None:
        """
        Abstract method that must be implemented by subclasses to plot data accuracy.

        Parameters:
        -----------
        Hist_data: object
            The object containing the accuracy data to be plotted.

        Returns:
        --------
        None
        """
        pass

    