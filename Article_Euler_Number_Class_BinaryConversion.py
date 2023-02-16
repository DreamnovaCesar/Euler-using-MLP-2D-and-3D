from Article_Euler_Number_Utilities import Utilities

import numpy as np
import json

# ? A class that provides functionality to convert decimal numbers to binary and store them in a list.
class BinaryConversion(Utilities):
    """
    A class that provides functionality to convert decimal numbers to binary and store them in a list.

    Args:
        Iter (int): The number of iterations for the binary conversion.
        ndim (str, optional): The dimensionality of the output list. Defaults to '2D'.

    Attributes:
        Iter (int): The number of iterations for the binary conversion.
        ndim (str): The dimensionality of the output list.

    Methods:
        __init__(self, Iter: int, ndim: str = '2D') -> None:
            Initializes the BinaryConversion object with the given number of iterations and dimensionality.
        __repr__(self):
            Returns a string representation of the BinaryConversion object.
        __str__(self):
            Returns a string description of the BinaryConversion object.
        __del__(self):
            Destroys the BinaryConversion object and prints a message to the console.
        data_dic(self):
            Returns a dictionary of the BinaryConversion object's attributes.
        create_json_file(self):
            Creates a JSON file with the BinaryConversion object's attributes and saves it to a specified file path.
        decimal_to_binary(Decimal_value: int) -> str:
            Converts a decimal number to binary and returns the binary string.
        decimal_to_binary_list(self, Number_iter: int, ndim: str = '2D') -> np.ndarray:
            Converts decimal numbers up to the given iteration count to binary and stores them in a list of arrays.

    Usage:
        Instantiate the BinaryConversion object with the desired number of iterations and dimensionality.
        Call the decimal_to_binary_list() method to convert decimal numbers up to the iteration count to binary and store
        them in a list of arrays.

    Example:
        >>> bc = BinaryConversion(5, ndim='3D')
        >>> bc.decimal_to_binary_list(5, ndim='3D')
        array([[[['0', '0'],
                 ['0', '1']],

                [['0', '1'],
                 ['0', '0']]],


               [[['0', '1'],
                 ['0', '1']],

                [['0', '1'],
                 ['1', '0']]],


               [[['1', '0'],
                 ['0', '0']],

                [['1', '0'],
                 ['0', '1']]],


               [[['1', '0'],
                 ['1', '0']],

                [['1', '0'],
                 ['1', '1']]],


               [[['1', '1'],
                 ['0', '0']],

                [['1', '1'],
                 ['0', '1']]]])
    """

    # * Initializing (Constructor)
    def __init__(self, Iter: int, ndim: str = '2D') -> None:
        """
        Initialize a new instance of the BinaryConversion class.

        Args:
            Iter (int): The maximum number of iterations to perform.
            ndim (str): The number of dimensions in the output list, either '2D' or '3D'. Defaults to '2D'.
        """

        # * General parameters
        self.Iter = Iter
        self.ndim = ndim


    # * Class variables
    def __repr__(self):
        """
        Returns the string representation of this BinaryConversion instance.

        Returns:
            str: The string representation of this BinaryConversion instance.
        """
        return f'''[{self.Iter}, 
                    {self.ndim}]''';

    # * Class description
    def __str__(self):
        """
        Returns a string representation of this BinaryConversion instance.

        Returns:
            str: A string representation of this BinaryConversion instance.
        """
        return  f'A class that provides functionality to convert decimal numbers to binary and store them in a list.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
        """
        Destructor that is called when this BinaryConversion instance is destroyed.
        """
        print('Destructor called, Euler number class destroyed.');

    # * Get data from a dic
    def data_dic(self):
        """
        Returns a dictionary containing the instance's Iter and ndim values.

        Returns:
            dict: A dictionary containing the instance's Iter and ndim values.
        """
        return {'Iter': str(self.Iter),
                'ndim': str(self.ndim)
                };
    
    def create_json_file(self):
        """
        Creates a JSON file with the given data and saves it to the specified file path.

        Returns:
        None
        """
        Data = {'Iter': str(self.Iter),
                'ndim': str(self.ndim)
                };

        with open('JSON documents', 'w') as file:
            json.dump(Data, file)
    # ?
    @staticmethod
    def decimal_to_binary(Decimal_value: int) -> str:
        """
        Convert a decimal value to its binary representation. prefix(0b)

        Args:
            Decimal_value (int): The decimal value to convert to binary.

        Returns:
            int: The binary representation of Decimal_value as an integer.
        """

        # * Conversion to int and binary
        Decimal_value = int(Decimal_value)
        Binary_value = format(Decimal_value, '08b')

        print('\n')
        print(Binary_value)

        return Binary_value

    # ?
    def decimal_to_binary_list(self, Number_iter: int, ndim: str = '2D') -> None:
        """
        Convert a range of decimal values to binary and return them as a list.

        Args:
            Number_iter (int): The maximum decimal value to convert to binary.
            ndim (str): The number of dimensions in the output list, either '2D' or '3D'. Defaults to '2D'.

        Returns:
            np.ndarray: A numpy array containing the binary representations of the decimal values.
        """
        
        # * Convert the input parameter to an integer.
        Number_iter = int(Number_iter)

        # * Create an empty list to store the reshape binary numbers.
        Qs = []

        if(ndim == '2D'):
            
            # * If `ndim` is '2D', create a 2D binary array for each decimal number.
            for i in range(Number_iter):

                # * Convert the decimal number to binary using the `decimal_to_binary` method.
                Binary_value = self.decimal_to_binary(i)

                # * Reshape the resulting binary string as a 2D numpy array with dimensions (2, 2).
                Array = np.reshape(Binary_value, (2, 2))

                # * Convert the numpy array to a list and add it to the list of binary arrays.
                Array_list = Array.tolist()
                Qs.append(Array_list)

            # * Convert the list of binary arrays to a numpy array and return it.
            Qs = np.array(Qs)
            return Qs
        
        elif(ndim == '3D'): 

            # * If `ndim` is '3D', create a 3D binary array for each decimal number.     
            for i in range(Number_iter):

                # * Convert the decimal number to binary using the `decimal_to_binary` method.
                Binary_value = self.decimal_to_binary(i)

                # * Reshape the resulting binary string as a 3D numpy array with dimensions (2, 2, 2).
                Array = np.reshape(Binary_value, (2, 2, 2))
                # * Convert the numpy array to a list and add it to the list of binary arrays.
                Array_list = Array.tolist()
                Qs.append(Array_list)

            # * Convert the list of binary arrays to a numpy array and return it.
            Qs = np.array(Qs)
            return Qs