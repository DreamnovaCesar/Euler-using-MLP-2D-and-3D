import numpy as np

# Function to convert decimal to binary
# using built-in python function

def assign_value(Tuple_index, Tuple_value):

    Output = np.zeros((256), dtype = 'int')

    OutList = []
    #print(Output.dtype)
    #print(len(Output))
    print(len(Tuple_index))
    print(len(Tuple_value))

    for i, element in enumerate(Output):
        for j, value in enumerate(Tuple_index):
            if Tuple_index[j] == i:

                Output[i] = Tuple_value[j]
            else:
                pass

    print(Output)  
    print(Output.dtype) 

    print()

    return Output



def decimalToBinary(n):
    # converting decimal to binary
    # and removing the prefix(0b)
    value = format(n, '08b')
    return value
   
# Driver code
if __name__ == '__main__':
    # calling function
    # with decimal argument
    label = 0
    
    Tuple_index = (2, 9, 11, 24, 25, 26, 27, 33, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 129, 131, 137, 139, 148, 149, 150, 151, 156, 157, 158, 159, 161, 163, 169, 171, 180, 181, 182, 183, 188, 189, 190, 191)
    #Tuple_value = (1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1)
    Tuple_value = (1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)
    Array_prediction = np.zeros((2, 4), dtype = 'int')
    #print(Array_prediction)

    assign_value(Tuple_index, Tuple_value)

    for i in range(256):
        value = decimalToBinary(i)

        Array_prediction[1][3] = value[7]   # h
        Array_prediction[1][2] = value[6]   # g
        Array_prediction[1][1] = value[5]   # f
        Array_prediction[1][0] = value[4]   # e

        Array_prediction[0][3] = value[3]   # d
        Array_prediction[0][2] = value[2]   # c
        Array_prediction[0][1] = value[1]   # b
        Array_prediction[0][0] = value[0]   # a


        #print('[{}, {}, {}, {}, {}, {}, {}, {}],'.format(value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7]))
        #print('{}{}{}{}{}{}{}{}, {}'.format(value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7], label))

        