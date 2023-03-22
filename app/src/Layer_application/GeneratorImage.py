import numpy as np

from .Generator import Generator

class GeneratorImage(Generator):
    """
    A class for generating 2D images.

    Parameters
    ----------
    Prob_0 : float
        The probability of generating a 0 value in the image.
    Prob_1 : float
        The probability of generating a 1 value in the image.
    _Height : int
        The height of the generated image.
    _Width : int
        The width of the generated image.

    Returns
    -------
    Data_2D : ndarray
        A 2D numpy array representing the generated image.
    """

    def generator(
        Prob_0 : float, 
        Prob_1 : float,
        _Height : int,
        _Width : int
    ) -> np.ndarray:
        """
        Generate a 2D image with the specified probabilities and dimensions.

        Parameters
        ----------
        Prob_0 : float
            The probability of generating a 0 value in the image.
        Prob_1 : float
            The probability of generating a 1 value in the image.
        _Height : int
            The height of the generated image.
        _Width : int
            The width of the generated image.

        Returns
        -------
        Data_2D : ndarray
            A 2D numpy array representing the generated image.
        """     

        # * Generate a 1D array of random integers with values 0 or 1, where the 
        # * probability of 0 and 1 are given by Prob_0 and Prob_1 respectively
        Data_2D = np.random.choice(2, _Height * _Width, p = [Prob_0, Prob_1]);
        Data_2D = Data_2D.reshape(_Height, _Width);

        # * Create a new 2D array with shape (_Height+2, _Width+2) filled with zeros
        Data_2D_edges = np.zeros((Data_2D.shape[0] + 2, 
                                  Data_2D.shape[1] + 2))
        
        # * Copy the values of Data_2D into the central portion of Data_2D_edges
        Data_2D_edges[1:Data_2D_edges.shape[0] - 1, 
                      1:Data_2D_edges.shape[1] - 1] = Data_2D

        # * Return the resulting 2D array
        return Data_2D_edges