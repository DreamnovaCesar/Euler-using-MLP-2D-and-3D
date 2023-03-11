import numpy as np

from typing import List

from .ConvertionDecimalBinaryByte import ConvertionDecimalBinaryByte
from .ConvertionDecimalBinaryNibble import ConvertionDecimalBinaryNibble

from typing import Union

class BinaryStorageList(object):
    """
    A class for creating a list of binary arrays and converting it to a numpy array.

    Methods
    -------
    to_numpy_array() -> np.ndarray:
        Convert a list of binary arrays to a numpy array.

    """

    def __init__(self, Convertion_object : Union[ConvertionDecimalBinaryByte, ConvertionDecimalBinaryNibble]) -> None:
        """
        Initialize the BinaryStorageList object.

        Parameters
        ----------
        Convertion_object : object
            An object of a class that implements the `convertion_system` method.

        """

        self.Convertion_object = Convertion_object;

        
    def to_numpy_array(self) -> np.ndarray:
        """
        Convert a list of binary arrays to a numpy array.

        Returns
        -------
        np.ndarray
            A numpy array containing the binary arrays.

        """

        # * Create an empty list called Storage_list
        Storage_list = []

        # * Check if the Convertion_object is an instance of ConvertionDecimalBinaryNibble
        if(self.Convertion_object == ConvertionDecimalBinaryNibble):
            Binary_number = '10000';
            Decimal_number = int(Binary_number, 2);

        # * Check if the Convertion_object is an instance of ConvertionDecimalBinaryByte
        if(self.Convertion_object == ConvertionDecimalBinaryByte):
            Binary_number = '100000000';
            Decimal_number = int(Binary_number, 2);
        
        # * Loop through the range of numbers from 0 to the Decimal_number - 1
        for i in range(Decimal_number):

            # * Call the convertion_system method of the Convertion_object and assign the result to the variable Binary_array
            Binary_array = self.Convertion_object.convertion_system(i);

            # * Convert the Binary_array to a list
            Binary_array = Binary_array.tolist();
            Storage_list.append(Binary_array);

        # * Convert the list of binary arrays to a numpy array and return it.
        Storage_list = np.array(Storage_list);

        return Storage_list;

