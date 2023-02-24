import numpy as np
import matplotlib.pyplot as plt
from Article_Euler_Number_Utilities import Utilities

class ArrayVisualization:
    """
    A class that contains methods to visualize 3D arrays.
    """
    @staticmethod
    def show_array_matplotlib(Image: str) -> np.ndarray:
        """
        Display a 2D numpy array using matplotlib.

        Parameters
        ----------
        Image : str
            The path to the input data file.

        Returns
        -------
        np.ndarray
            The input data as a 2D numpy array.
        """

        # * load the data from the file
        #Data = np.genfromtxt(Image, delimiter = ",")
        Array = np.loadtxt(Image, delimiter = ',')

        print(Array)
        # * Convert the array to integers (if it's not already) for consistency.
        Array = Array.astype(int)

        # * reshape the data to 2D
        Array = Array.reshape(Array.shape[0], Array.shape[1]);

        # * display the 2D array using voxels
        plt.imshow(Array, cmap = 'gray', interpolation = 'nearest')
        plt.show()
        plt.close()

        return Array
    
    @staticmethod
    def show_array_matplotlib_3D(Object: str) -> np.ndarray:
        """
        Display a 3D numpy array using matplotlib.

        Parameters
        ----------
        Object : str
            The path to the input data file.

        Returns
        -------
        np.ndarray
            The input data as a 3D numpy array.
        """

        # * load the data from the file
        Array = np.loadtxt(Object, delimiter = ',')

        # * Convert the array to integers (if it's not already) for consistency.
        Array = Array.astype(int)

        # * reshape the data to 3D
        Height = Array.shape[0]/Array.shape[1]
        Array = Array.reshape((int(Height), int(Array.shape[1]), int(Array.shape[1])))

        # * display the 3D array using voxels
        ax = plt.figure().add_subplot(projection = '3d')
        ax.voxels(Array, edgecolors = 'gray')
        plt.show()
        plt.close()

        return Array
    