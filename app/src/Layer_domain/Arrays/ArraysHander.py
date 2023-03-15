import numpy as np

from .ArrayHandler import ArrayHandler

class ArraysHandlder(ArrayHandler):
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
                 Object: str
                 ):
        """
        Initialize OctovoxelHandler.

        Parameters:
        ----------
        Arrays : np.ndarray
            An array of numbers.

        Storage_list : List[str]
            A list of strings containing binary values.
        """
        self.Object = Object;

    def get_array(self) -> np.ndarray:
        """
        Calculate Combinations_int based on the given Storage_list and return them as a numpy array.
        """

        # Create an empty list to store the resulting 1D arrays
        Arrays = []

        # * If `_ndim` is set to "3D", the array is reshaped to 3D and printed.
        Height = self.Object.shape[0]/self.Object.shape[1]
        Array = self.Object.reshape(int(Height), int(self.Object.shape[1]), int(self.Object.shape[1]))
        
        # Create two zero-filled arrays - one with shape (l, l, l) and the other with shape (8)
        #Array_prediction_octov = np.zeros((l, l, l))
        Array_prediction = np.zeros((8))

        # * Initial size of the octovovel.
        Octovoxel_size = 2

        # Iterate over each pixel in the 3D array
        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):
                for k in range(Array.shape[2] - 1):

                    #Array[i:l + i, j:l + j, k:l + k]

                    # Extract the values of the surrounding voxels to create an array of predictions
                    Array_prediction[0] = Array[i + 1][j][k]
                    Array_prediction[1] = Array[i + 1][j][k + 1]

                    Array_prediction[2] = Array[i][j][k]
                    Array_prediction[3] = Array[i][j][k + 1]

                    Array_prediction[4] = Array[i + 1][j + 1][k]
                    Array_prediction[5] = Array[i + 1][j + 1][k + 1]

                    Array_prediction[6] = Array[i][j + 1][k]
                    Array_prediction[7] = Array[i][j + 1][k + 1]
                    print('\n')

                    Array_prediction_list = Array_prediction.tolist()
                    Array_prediction_list_int = [int(i) for i in Array_prediction_list]
                    print("Kernel array")

                    print(Array[i:Octovoxel_size + i, 
                                j:Octovoxel_size + j, 
                                k:Octovoxel_size + k])
                    
                    print('\n')
                    print("Prediction array")
                    print(Array_prediction)
                    print('\n')
                    Arrays.append(Array_prediction_list_int)
                    print('\n')

        # Display the resulting list of 1D arrays
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays