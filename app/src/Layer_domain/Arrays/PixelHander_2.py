import numpy as np

from .ArrayHandler import ArrayHandler

class PixelHandler(ArrayHandler):
    """
    Handler for pixel data in an array.

    Parameters
    ----------
    Arrays : np.ndarray
        Numpy array containing the pixel data.
    Storage_list : list[str]
        List of binary combinations to be searched in the array.

    Returns
    -------
    Combinations_int : np.ndarray
        An array containing the number of occurrences of the binary combinations in the given pixel data array.

    Notes
    -----
    The array of pixel data is searched in Quadra_size of size 2. The binary combinations in the storage list are compared
    with each Pixel and the number of occurrences of each combination is recorded in the output array.
    """
    def __init__(self, 
                 Arrays: np.ndarray, 
                 Storage_list: list[str]):
        """
        Initialize PixelHandler.

        Parameters:
        ----------
        Arrays : np.ndarray
            An array of numbers.

        Storage_list : List[str]
            A list of strings containing binary values.
        """

        self.Arrays = Arrays;
        self.Storage_list = Storage_list;

    def get_array(self) -> np.ndarray:
        """
        Calculate Combinations_int based on the given Storage_list and return them as a numpy array.
        """
        # * Calculate the combinations using broadcasting and NumPy functions.
        quadrapixels = np.lib.stride_tricks.sliding_window_view(self.Arrays, window_shape=(2,2))
        in_storage_list = np.isin(quadrapixels, self.Storage_list).any(axis=(2,3))
        combinations_int = np.count_nonzero(in_storage_list, axis=1)
        
        # * Return the calculated combinations as a numpy array.
        return combinations_int