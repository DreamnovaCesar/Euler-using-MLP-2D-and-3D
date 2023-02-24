import numpy as np

from .Class_ArrayHandler import ArrayHandler

class OctovoxelHandler(ArrayHandler):
    def __init__(self, Arrays: np.ndarray, Qs: list[str]):
        self.Arrays = Arrays
        self.Qs = Qs

    def get_number(self) -> np.ndarray:
        
        # * If `_ndim` is set to "3D", the array is reshaped to 3D and printed.
        Height = self.Arrays.shape[0]/self.Arrays.shape[1]
        self.Arrays = self.Arrays.reshape(int(Height), int(self.Arrays.shape[1]), int(self.Arrays.shape[1]))

        Octovoxel_size = 2
        Binary_number = '100000000'
        Combinations = int(Binary_number, 2)
        
        q_values = np.zeros((Combinations), dtype='int')
        for i in range(self.Arrays.shape[0] - 1):
            for j in range(self.Arrays.shape[1] - 1):
                for k in range(self.Arrays.shape[2] - 1):
                    for index in range(len(self.Qs)):
                        if np.array_equal(np.array(self.Arrays[i:Octovoxel_size + i, j:Octovoxel_size + j, k:Octovoxel_size + k]), np.array(self.Qs[index])):
                            q_values[index] += 1

        return q_values