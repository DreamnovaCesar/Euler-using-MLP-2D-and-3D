import numpy as np

from .ArrayHandler import ArrayHandler

class OctovoxelHandler(ArrayHandler):
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

        # * If `_ndim` is set to "3D", the array is reshaped to 3D and printed.
        Height = self.Arrays.shape[0]/self.Arrays.shape[1];
        # * Reshape the array to a 3D array based on the calculated height.
        self.Arrays = self.Arrays.reshape(int(Height), int(self.Arrays.shape[1]), int(self.Arrays.shape[1]));

        # * Set the size of octovoxel and calculate the combinations.
        Octovoxel_size = 2;
        Binary_number = '100000000';
        Combinations = int(Binary_number, 2);
        
        # * Create an array to store Combinations_int and initialize it with zeros.
        Combinations_int = np.zeros((Combinations), dtype='int');
        for i in range(self.Arrays.shape[0] - 1):
            for j in range(self.Arrays.shape[1] - 1):
                for k in range(self.Arrays.shape[2] - 1):
                    for index in range(len(self.Storage_list)):
                        if np.array_equal(np.array(self.Arrays[i:Octovoxel_size + i, j:Octovoxel_size + j, k:Octovoxel_size + k]), 
                                          np.array(self.Storage_list[index])):
                            
                            Combinations_int[index] += 1;

        # * Return the calculated Combinations_int as a numpy array.
        return Combinations_int