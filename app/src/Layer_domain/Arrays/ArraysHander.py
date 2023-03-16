import numpy as np

from .ArrayHandler import ArrayHandler

class ArraysHandlder(ArrayHandler):
    """
    Class that handles arrays and returns a list of 1D arrays.

    Parameters
    ----------
    Object : str
        A numpy array.

    Methods
    -------
    get_array() -> list:
        Returns a list of 1D arrays.
    """

    def __init__(self, 
                 Object: str
                 ):
        """
        Initializes an instance of the ArraysHandler class.
        
        Parameters
        ----------
        Object : str
            A numpy array.
        """

        self.Object = Object;

    def get_array(self) -> list:
        """
        Returns a list of 1D arrays based on the input numpy array.

        Returns
        -------
        list
            A list of 1D arrays.
        """
       
        # * Create an empty list to store the resulting 1D arrays
        Arrays = []

        # * If `_ndim` is set to "3D", the array is reshaped to 3D and printed.
        Height = self.Object.shape[0]/self.Object.shape[1]
        # * Reshape the array to a 3D array based on the calculated height.
        self.Array = self.Object.reshape(int(Height), int(self.Object.shape[1]), int(self.Object.shape[1]))
        
        # * Create two zero-filled arrays - one with shape (l, l, l) and the other with shape (8)
        Array_prediction = np.zeros((8))

        # * Initial size of the octovovel.
        Octovoxel_size = 2

        # Iterate over each pixel in the 3D array
        for i in range(self.Array.shape[0] - 1):
            for j in range(self.Array.shape[1] - 1):
                for k in range(self.Array.shape[2] - 1):

                    # Extract the values of the surrounding voxels to create an array of predictions
                    Array_prediction[0] = self.Array[i + 1][j][k]
                    Array_prediction[1] = self.Array[i + 1][j][k + 1]

                    Array_prediction[2] = self.Array[i][j][k]
                    Array_prediction[3] = self.Array[i][j][k + 1]

                    Array_prediction[4] = self.Array[i + 1][j + 1][k]
                    Array_prediction[5] = self.Array[i + 1][j + 1][k + 1]

                    Array_prediction[6] = self.Array[i][j + 1][k]
                    Array_prediction[7] = self.Array[i][j + 1][k + 1]

                    # * Append the array of predictions to the list
                    Arrays.append(Array_prediction.astype(int).tolist())

                    #Array_prediction_list = Array_prediction.tolist()
                    #Array_prediction_list_int = [int(i) for i in Array_prediction_list]
                    #Arrays.append(Array_prediction_list_int)
                    #print("Kernel array")

                    #print(self.Array[i:Octovoxel_size + i, 
                    #                 j:Octovoxel_size + j, 
                    #                 k:Octovoxel_size + k])
                    
                    #print('\n')
                    #print("Prediction array")
                    #print(Array_prediction)
                    #print('\n')
    

        '''for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')'''
        
        return Arrays