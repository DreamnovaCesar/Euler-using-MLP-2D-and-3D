import matplotlib.pyplot as plt

from Class_ArrayVisualization import ArrayVisualization

class Array3DVisualization(ArrayVisualization):
    """
    A class that contains methods to visualize 3D arrays.
    """

    def show_array(self, data):
        """
        Display a 3D numpy array using matplotlib.

        Parameters
        ----------
        data : np.ndarray
            The input data as a 3D numpy array.

        Returns
        -------
        np.ndarray
            The input data as a 3D numpy array.
        """
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(data, edgecolors='gray')
        plt.show()
        plt.close()