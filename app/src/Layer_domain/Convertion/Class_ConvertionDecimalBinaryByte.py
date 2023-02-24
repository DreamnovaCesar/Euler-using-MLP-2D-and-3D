import numpy as np

from .Class_ConversationSystem import ConvertionSystem

class ConvertionDecimalBinaryByte(ConvertionSystem):

    @staticmethod
    def convertion_system(Value_to_convert: int) -> np.ndarray:
        """
        Convert a decimal value to its binary representation. prefix(0b)

        Parameters
        ----------
        Value_to_convert : str 
            The decimal value to convert to binary.

        Returns
        -------
        int 
            The binary representation of Value_to_convert as an integer.
        """

        # * Conversion to int and binary
        Value_to_convert = int(Value_to_convert)
        Binary_value = format(Value_to_convert, '08b')

        Shape = (2,) * 3
        Binary_array = [int(x) for x in Binary_value]
        Binary_array = np.reshape([Binary_array], Shape)

        return Binary_array
    