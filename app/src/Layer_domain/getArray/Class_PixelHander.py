import numpy as np

from .Class_ArrayHandler import ArrayHandler

class PixelHandler(ArrayHandler):
    def __init__(self, Arrays: np.ndarray, Qs: list[str]):
        self.Arrays = Arrays
        self.Qs = Qs

    def get_number(self) -> np.ndarray:

        Octovoxel_size = 2
        Binary_number = '10000'
        Combinations = int(Binary_number, 2)

        q_values = np.zeros((Combinations), dtype='int')
        for i in range(self.Arrays.shape[0] - 1):
            for j in range(self.Arrays.shape[1] - 1):
                for index in range(len(self.Qs)):
                    if np.array_equal(np.array(self.Arrays[i:Octovoxel_size + i, j:Octovoxel_size + j]), np.array(self.Qs[index])):
                        q_values[index] += 1
                        
        return q_values