import matplotlib.pyplot as plt

from ArrayVisualization import ArrayVisualization

class Array2DVisualization(ArrayVisualization):
    """
    A class that contains methods to visualize 2D arrays.
    """
    
    def show_array(self, data):
        """
        Display a 2D numpy array using matplotlib.

        Parameters
        ----------
        data : np.ndarray
            The input data as a 2D numpy array.

        Returns
        -------
        np.ndarray
            The input data as a 2D numpy array.
        """
        plt.imshow(data, cmap='gray', interpolation='nearest')
        plt.show()
        plt.close()