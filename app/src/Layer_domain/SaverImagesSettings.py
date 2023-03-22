import os
import numpy as np
import matplotlib.pyplot as plt

from .Saver import Saver

class SaverImagesSettings(Saver):
    """
    A class that inherits from the `Saver` base class and saves 2D edge data with Euler number as images.

    Parameters
    ----------
    Saver : `Saver` object
        The base class that SaverImagesSettings is inheriting from.

    Methods
    -------
    save_file(i, Folder_path, Euler_number, Data_2D_edges)
        Save the 2D edge data as a .png image with a given file name and file path, along with the Euler number.

    Attributes
    ----------
    None

    """

    def save_file(
        self, 
        i : int, 
        Folder_path: str, 
        Euler_number : int, 
        Data_2D_edges : np.ndarray, 
    ):
        """
        Save 2D edge data as a .png image with a given file name and file path, along with the Euler number.

        Parameters
        ----------
        i : int
            The index of the file.
        Folder_path : str
            The folder path where the image file is to be saved.
        Euler_number : int
            The Euler number to be displayed in the image title.
        Data_2D_edges : np.ndarray
            The 2D edge data to be saved as a .png image.

        Returns
        -------
        None
        """

        Image_name = "Image_with_euler_{}_2D.png".format(i)
        Image_path = os.path.join(Folder_path, Image_name)
        plt.title('Euler: {}'.format(Euler_number))
        plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest')
        plt.savefig(Image_path)
        plt.close()