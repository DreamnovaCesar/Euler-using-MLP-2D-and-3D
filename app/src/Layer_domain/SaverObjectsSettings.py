import os
import numpy as np
import matplotlib.pyplot as plt

from .Saver import Saver

class SaverObjectsSettings(Saver):
    """
    A class that saves 3D object slices as images in a given folder path.

    Parameters
    ----------
    Folder_path : str
        The path of the folder where to save the image.
    Euler_number : int
        The Euler number of the 3D object slice.
    Data_3D_edges : numpy.ndarray
        A 3D numpy array of the object edges data.
    i : int
        The i-th index of the 3D array slice to save.
    j : int
        The j-th index of the 3D array slice to save.

    Methods
    -------
    save_file(Folder_path: str, Euler_number: int, Data_3D_edges: np.ndarray, i: int, j: int)
        Saves a 3D object slice as an image with a title and a given name in the given folder path.

    """

    def save_file(
        self, 
        i : int, 
        j : int,
        Folder_path: str, 
        Euler_number : int, 
        Data_3D_edges : np.ndarray, 
    ):
        """
        Saves a 3D object slice as an image with a title and a given name in the given folder path.

        Parameters
        ----------
        Folder_path : str
            The path of the folder where to save the image.
        Euler_number : int
            The Euler number of the 3D object slice.
        Data_3D_edges : numpy.ndarray
            A 3D numpy array of the object edges data.
        i : int
            The i-th index of the 3D array slice to save.
        j : int
            The j-th index of the 3D array slice to save.
        """

        Image_name = "Image_slice_random_{}_{}_3D".format(i, j)
        Image_path = os.path.join(Folder_path, Image_name)
        #plt.title('P_0: {}, P_1: {}'.format(P_0, P_1))
        plt.title('Euler: {}'.format(Euler_number))
        plt.imshow(Data_3D_edges[j], cmap = 'gray', interpolation = 'nearest')
        plt.savefig(Image_path)
        plt.close()