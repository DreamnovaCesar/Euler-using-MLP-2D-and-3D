import numpy as np

from .ArrayHandler import ArrayHandler

class OctovoxelHandler_2(ArrayHandler):
    """
    A class for handling octovoxels and obtaining q_values based on the given Qs.

    Attributes:
    -----------
    Arrays : np.ndarray
        An array of numbers.

    Storage_list : List[str]
        A list of strings containing binary values.

    Methods:
    --------
    get_number() -> np.ndarray:
        Calculate q_values based on the given Storage_list and returns them as a numpy array.
    """
    
    def __init__(self, 
                 Arrays: np.ndarray, 
                 Storage_list: list[str]):
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

    def get_array(self) -> np.ndarray:
        """
        Calculate Combinations_int based on the given Storage_list and return them as a numpy array.
        """
        # * Reshape the array to a 3D array based on the calculated height.
        height = self.Arrays.shape[0] // self.Arrays.shape[1]
        arrays_3d = self.Arrays.reshape(height, self.Arrays.shape[1], self.Arrays.shape[1])
        
        # * Calculate the combinations using broadcasting and NumPy functions.
        octovoxels = np.lib.stride_tricks.sliding_window_view(arrays_3d, window_shape=(2,2,2))
        in_storage_list = np.isin(octovoxels, self.Storage_list).any(axis=(3,4,5))
        combinations_int = np.bincount(np.flatnonzero(in_storage_list), minlength=len(self.Storage_list))
        
        # * Return the calculated combinations as a numpy array.
        return combinations_int