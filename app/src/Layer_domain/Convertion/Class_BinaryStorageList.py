import numpy as np

from typing import List

from .Class_ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from .Class_ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble

class BinaryStorageList():
    def __init__(self, Convertion_object : object) -> None:
        self.Convertion_object = Convertion_object
    
    def to_numpy_array(self) -> np.ndarray:
        
        Qs = []

        if(self.Convertion_object == ConvertionDecimalBinaryNibble):
            Binary_number = '10000'
            Decimal_number = int(Binary_number, 2)

        if(self.Convertion_object == ConvertionDecimalBinaryByte):
            Binary_number = '100000000'
            Decimal_number = int(Binary_number, 2)

        for i in range(Decimal_number):

            Binary_array = self.Convertion_object.convertion_system(i)

            Binary_array = Binary_array.tolist()
            Qs.append(Binary_array)

        # * Convert the list of binary arrays to a numpy array and return it.
        Qs = np.array(Qs)
        return Qs

