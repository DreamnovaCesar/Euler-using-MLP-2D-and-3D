import os
import numpy as np
import matplotlib.pyplot as plt

from .Saver import Saver

class SaverImagesRandomly(Saver):
    """
    A class that inherits from the `Saver` base class and saves 2D edge data as images.

    Parameters
    ----------
    Saver : `Saver` object
        The base class that SaverImagesRandomly is inheriting from.

    Methods
    -------
    save_file(i, Prob_0, Prob_1, Folder_path, Data_2D_edges)
        Save the 2D edge data as a .png image with a given file name and file path.

    Attributes
    ----------
    None

    """

    def save_file(
        self, 
        i : int, 
        Prob_0 : float,
        Prob_1 : float,
        Folder_path: str, 
        Data_2D_edges : np.ndarray, 
    ):
        """
        Save 2D edge data as a .png image with a given file name and file path.

        Parameters
        ----------
        i : int
            The index of the file.
        Prob_0 : float
            The probability of state 0.
        Prob_1 : float
            The probability of state 1.
        Folder_path : str
            The folder path where the image file is to be saved.
        Data_2D_edges : np.ndarray
            The 2D edge data to be saved as a .png image.

        Returns
        -------
        None
        """
       
        Image_name = "Image_random_{}_2D.png".format(i);
        Image_path = os.path.join(Folder_path, Image_name);
        plt.title('P_0: {}, P_1: {}'.format(Prob_0, Prob_1));
        plt.imshow(Data_2D_edges, cmap = 'gray', interpolation = 'nearest');
        plt.savefig(Image_path);
        plt.close();
