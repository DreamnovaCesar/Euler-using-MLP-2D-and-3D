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
    with each Octovoxel and the number of occurrences of each combination is recorded in the output array.
    """
    def __init__(self, Arrays: np.ndarray, Storage_list: list[str]):
        """
        Initialize OctovoxelHandler.

        Parameters:
        ----------
        Arrays : np.ndarray
            An array of numbers.

        Storage_list : List[str]
            A list of strings containing binary values.
        """

        self.Arrays = Arrays;
        self.Storage_list = Storage_list;

    def get_number(self) -> np.ndarray:
        """
        Calculate Combinations_int based on the given Storage_list and return them as a numpy array.
        """

        # * Set the size of quadrapixel and calculate the combinations.
        Quadra_size = 2;
        Binary_number = '10000';
        Combinations = int(Binary_number, 2);

        # * Create an array to store Combinations_int and initialize it with zeros.
        Combinations_int = np.zeros((Combinations), dtype='int');
        for i in range(self.Arrays.shape[0] - 1):
            for j in range(self.Arrays.shape[1] - 1):
                for index in range(len(self.Storage_list)):
                    if np.array_equal(np.array(self.Arrays[i:Quadra_size + i, j:Quadra_size + j]), 
                                      np.array(self.Storage_list[index])):
                        
                        Combinations_int[index] += 1;
        
        # * Return the calculated Combinations_int as a numpy array.                
        return Combinations_int