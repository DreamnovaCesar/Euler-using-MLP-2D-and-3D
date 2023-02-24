from Article_Euler_Number_Utilities import Utilities

import numpy as np

# ? A class that provides functionality to convert decimal numbers to binary and store them in a list.
class BinaryConversion():
 
    # * Initializing (Constructor)
    def __init__(self, Iter: int, ndim: str = '2D') -> None:
    

        # * General parameters
        self.Iter = Iter
        self.ndim = ndim


    # * Class variables
    def __repr__(self):
      
        return f'''[{self.Iter}, 
                    {self.ndim}]''';

    # * Class description
    def __str__(self):
      
        return  f'A class that provides functionality to convert decimal numbers to binary and store them in a list.';
    
    # * Deleting (Calling destructor)
    def __del__(self):
     
        print('Destructor called, Euler number class destroyed.');

    # * Get data from a dic
    def data_dic(self):
        
        return {'Iter': str(self.Iter),
                'ndim': str(self.ndim)
                };
    
    @staticmethod
    def decimal_to_binary(Decimal_value: int) -> str:
      

        # * Conversion to int and binary
        Decimal_value = int(Decimal_value)
        Binary_value = format(Decimal_value, '08b')

        #print('\n')
        #print(Binary_value)

        return Binary_value

    # ?
    def decimal_to_binary_list(self) -> None:
     
        
        # * Convert the input parameter to an integer.
        Number_iter = int(self.Iter)

        # * Create an empty list to store the reshape binary numbers.
        Qs = []

        if(self.ndim == '2D'):
            
            # * If `ndim` is '2D', create a 2D binary array for each decimal number.
            for i in range(Number_iter):

                # * Convert the decimal number to binary using the `decimal_to_binary` method.
                Binary_value = self.decimal_to_binary(i)
                Binary_value = [int(x) for x in Binary_value]
                
                # * Reshape the resulting binary string as a 2D numpy array with dimensions (2, 2).
                Array = np.reshape([Binary_value], (2, 2))

                # * Convert the numpy array to a list and add it to the list of binary arrays.
                Array_list = Array.tolist()
                Qs.append(Array_list)

            # * Convert the list of binary arrays to a numpy array and return it.
            Qs = np.array(Qs)
            return Qs
        
        elif(self.ndim  == '3D'): 

            # * If `ndim` is '3D', create a 3D binary array for each decimal number.     
            for i in range(Number_iter):

                # * Convert the decimal number to binary using the `decimal_to_binary` method.
                Binary_value = self.decimal_to_binary(i)
                Binary_value = [int(x) for x in Binary_value]

                # * Reshape the resulting binary string as a 3D numpy array with dimensions (2, 2, 2).
                Array = np.reshape(Binary_value, (2, 2, 2))
                # * Convert the numpy array to a list and add it to the list of binary arrays.
                Array_list = Array.tolist()
                Qs.append(Array_list)

            # * Convert the list of binary arrays to a numpy array and return it.
            Qs = np.array(Qs)
            return Qs
        
a = BinaryConversion(20, '3D')

c = a.decimal_to_binary_list()

print(c[0])
