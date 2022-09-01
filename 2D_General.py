import numpy as np

def read_image_with_metadata(Array_file):
    """
    _summary_

    _extended_summary_

    Args:
        Array_file (_type_): _description_

    Returns:
        _type_: _description_
    """
    Array = np.loadtxt(Array_file, delimiter = ',')
    
    Array = Array.astype(int)

    print('\n')
    print('Array obtained')
    print('\n')
    print(Array)
    print('\n')
    print('Number of rows: {}'.format(Array.shape[0]))
    print('\n')
    print('Number of columns: {}'.format(Array.shape[1]))
    print('\n')
    
    return Array

Connectivity_4_first_array = np.array([     [ 1,  0],
                                            [ 0,  0]    ], dtype = 'int')

Connectivity_4_second_array = np.array([    [ 1,  1],
                                            [ 1,  0]    ], dtype = 'int')

Connectivity_4_third_array = np.array([     [ 1,  0],
                                            [ 0,  1]    ], dtype = 'int')

Connectivity_8_first_array = np.array([     [ 1,  0],
                                            [ 0,  0]    ], dtype = 'int')

Connectivity_8_second_array = np.array([    [ 1,  1],
                                            [ 1,  0]    ], dtype = 'int')

Connectivity_8_third_array = np.array([     [ 0,  1],
                                            [ 1,  0]    ], dtype = 'int')

Input_2D = np.array([   [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 1, 1],
                        [0, 1, 0, 0],
                        [0, 1, 0, 1],
                        [0, 1, 1, 0],
                        [0, 1, 1, 1],
                        [1, 0, 0, 0],
                        [1, 0, 0, 1],
                        [1, 0, 1, 0],
                        [1, 0, 1, 1],
                        [1, 1, 0, 0],
                        [1, 1, 0, 1],
                        [1, 1, 1, 0],
                        [1, 1, 1, 1]  ], dtype = int)

Output_2D_4_Connectivity = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, -1, 0], dtype = int)

Output_2D_8_Connectivity = np.array([0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0], dtype = int)