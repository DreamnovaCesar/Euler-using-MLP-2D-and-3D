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
    print('\n')
    print('Number of channel: {}'.format(Array.shape[0]))
    print('\n')
    print('Number of rows: {}'.format(Array.shape[0]))
    print('\n')
    print('Number of columns: {}'.format(Array.shape[1]))
    print('\n')
    
    return Array
                        # a, b, c, d    e, f, g, h
Input_3D = np.array([   [[0, 0, 0, 0], [0, 0, 0, 0]],
                        [[0, 0, 0, 0], [0, 0, 0, 1]],
                        [[0, 0, 0, 0], [0, 0, 1, 0]],
                        [[0, 0, 0, 0], [0, 0, 1, 1]],
                        [[0, 0, 0, 0], [0, 1, 0, 0]],
                        [[0, 0, 0, 0], [0, 1, 0, 1]],
                        [[0, 0, 0, 0], [0, 1, 1, 0]],
                        [[0, 0, 0, 0], [0, 1, 1, 1]],
                        [[0, 0, 0, 0], [1, 0, 0, 0]],
                        [[0, 0, 0, 0], [1, 0, 0, 1]],
                        [[0, 0, 0, 0], [1, 0, 1, 0]],
                        [[0, 0, 0, 0], [1, 0, 1, 1]],
                        [[0, 0, 0, 0], [1, 1, 0, 0]],
                        [[0, 0, 0, 0], [1, 1, 0, 1]],
                        [[0, 0, 0, 0], [1, 1, 1, 0]],
                        [[0, 0, 0, 0], [1, 1, 1, 1]],
                        [[0, 0, 0, 1], [0, 0, 0, 0]],
                        [[0, 0, 0, 1], [0, 0, 0, 1]],
                        [[0, 0, 0, 1], [0, 0, 1, 0]],
                        [[0, 0, 0, 1], [0, 0, 1, 1]],
                        [[0, 0, 0, 1], [0, 1, 0, 0]],
                        [[0, 0, 0, 1], [0, 1, 0, 1]],
                        [[0, 0, 0, 1], [0, 1, 1, 0]],
                        [[0, 0, 0, 1], [0, 1, 1, 1]],
                        [[0, 0, 0, 1], [1, 0, 0, 0]],
                        [[0, 0, 0, 1], [1, 0, 0, 1]],
                        [[0, 0, 0, 1], [1, 0, 1, 0]],
                        [[0, 0, 0, 1], [1, 0, 1, 1]],
                        [[0, 0, 0, 1], [1, 1, 0, 0]],
                        [[0, 0, 0, 1], [1, 1, 0, 1]],
                        [[0, 0, 0, 1], [1, 1, 1, 0]],
                        [[0, 0, 0, 1], [1, 1, 1, 1]],
                        [[0, 0, 1, 0], [0, 0, 0, 0]],
                        [[0, 0, 1, 0], [0, 0, 0, 1]],
                        [[0, 0, 1, 0], [0, 0, 1, 0]],
                        [[0, 0, 1, 0], [0, 0, 1, 1]],
                        [[0, 0, 1, 0], [0, 1, 0, 0]],
                        [[0, 0, 1, 0], [0, 1, 0, 1]],
                        [[0, 0, 1, 0], [0, 1, 1, 0]],
                        [[0, 0, 1, 0], [0, 1, 1, 1]],
                        [[0, 0, 1, 0], [1, 0, 0, 0]],
                        [[0, 0, 1, 0], [1, 0, 0, 1]],
                        [[0, 0, 1, 0], [1, 0, 1, 0]],
                        [[0, 0, 1, 0], [1, 0, 1, 1]],
                        [[0, 0, 1, 0], [1, 1, 0, 0]],
                        [[0, 0, 1, 0], [1, 1, 0, 1]],
                        [[0, 0, 1, 0], [1, 1, 1, 0]],
                        [[0, 0, 1, 0], [1, 1, 1, 1]],
                        [[0, 0, 1, 1], [0, 0, 0, 0]],
                        [[0, 0, 1, 1], [0, 0, 0, 1]],
                        [[0, 0, 1, 1], [0, 0, 1, 0]],
                        [[0, 0, 1, 1], [0, 0, 1, 1]],
                        [[0, 0, 1, 1], [0, 1, 0, 0]],
                        [[0, 0, 1, 1], [0, 1, 0, 1]],
                        [[0, 0, 1, 1], [0, 1, 1, 0]],
                        [[0, 0, 1, 1], [0, 1, 1, 1]],
                        [[0, 0, 1, 1], [1, 0, 0, 0]],
                        [[0, 0, 1, 1], [1, 0, 0, 1]],
                        [[0, 0, 1, 1], [1, 0, 1, 0]],
                        [[0, 0, 1, 1], [1, 0, 1, 1]],
                        [[0, 0, 1, 1], [1, 1, 0, 0]],
                        [[0, 0, 1, 1], [1, 1, 0, 1]],
                        [[0, 0, 1, 1], [1, 1, 1, 0]],
                        [[0, 0, 1, 1], [1, 1, 1, 1]],
                        [[0, 1, 0, 0], [0, 0, 0, 0]],
                        [[0, 1, 0, 0], [0, 0, 0, 1]],
                        [[0, 1, 0, 0], [0, 0, 1, 0]],
                        [[0, 1, 0, 0], [0, 0, 1, 1]],
                        [[0, 1, 0, 0], [0, 1, 0, 0]],
                        [[0, 1, 0, 0], [0, 1, 0, 1]],
                        [[0, 1, 0, 0], [0, 1, 1, 0]],
                        [[0, 1, 0, 0], [0, 1, 1, 1]],
                        [[0, 1, 0, 0], [1, 0, 0, 0]],
                        [[0, 1, 0, 0], [1, 0, 0, 1]],
                        [[0, 1, 0, 0], [1, 0, 1, 0]],
                        [[0, 1, 0, 0], [1, 0, 1, 1]],
                        [[0, 1, 0, 0], [1, 1, 0, 0]],
                        [[0, 1, 0, 0], [1, 1, 0, 1]],
                        [[0, 1, 0, 0], [1, 1, 1, 0]],
                        [[0, 1, 0, 0], [1, 1, 1, 1]],
                        [[0, 1, 0, 1], [0, 0, 0, 0]],
                        [[0, 1, 0, 1], [0, 0, 0, 1]],
                        [[0, 1, 0, 1], [0, 0, 1, 0]],
                        [[0, 1, 0, 1], [0, 0, 1, 1]],
                        [[0, 1, 0, 1], [0, 1, 0, 0]],
                        [[0, 1, 0, 1], [0, 1, 0, 1]],
                        [[0, 1, 0, 1], [0, 1, 1, 0]],
                        [[0, 1, 0, 1], [0, 1, 1, 1]],
                        [[0, 1, 0, 1], [1, 0, 0, 0]],
                        [[0, 1, 0, 1], [1, 0, 0, 1]],
                        [[0, 1, 0, 1], [1, 0, 1, 0]],
                        [[0, 1, 0, 1], [1, 0, 1, 1]],
                        [[0, 1, 0, 1], [1, 1, 0, 0]],
                        [[0, 1, 0, 1], [1, 1, 0, 1]],
                        [[0, 1, 0, 1], [1, 1, 1, 0]],
                        [[0, 1, 0, 1], [1, 1, 1, 1]],
                        [[0, 1, 1, 0], [0, 0, 0, 0]],
                        [[0, 1, 1, 0], [0, 0, 0, 1]],
                        [[0, 1, 1, 0], [0, 0, 1, 0]],
                        [[0, 1, 1, 0], [0, 0, 1, 1]],
                        [[0, 1, 1, 0], [0, 1, 0, 0]],
                        [[0, 1, 1, 0], [0, 1, 0, 1]],
                        [[0, 1, 1, 0], [0, 1, 1, 0]],
                        [[0, 1, 1, 0], [0, 1, 1, 1]],
                        [[0, 1, 1, 0], [1, 0, 0, 0]],
                        [[0, 1, 1, 0], [1, 0, 0, 1]],
                        [[0, 1, 1, 0], [1, 0, 1, 0]],
                        [[0, 1, 1, 0], [1, 0, 1, 1]],
                        [[0, 1, 1, 0], [1, 1, 0, 0]],
                        [[0, 1, 1, 0], [1, 1, 0, 1]],
                        [[0, 1, 1, 0], [1, 1, 1, 0]],
                        [[0, 1, 1, 0], [1, 1, 1, 1]],
                        [[0, 1, 1, 1], [0, 0, 0, 0]],
                        [[0, 1, 1, 1], [0, 0, 0, 1]],
                        [[0, 1, 1, 1], [0, 0, 1, 0]],
                        [[0, 1, 1, 1], [0, 0, 1, 1]],
                        [[0, 1, 1, 1], [0, 1, 0, 0]],
                        [[0, 1, 1, 1], [0, 1, 0, 1]],
                        [[0, 1, 1, 1], [0, 1, 1, 0]],
                        [[0, 1, 1, 1], [0, 1, 1, 1]],
                        [[0, 1, 1, 1], [1, 0, 0, 0]],
                        [[0, 1, 1, 1], [1, 0, 0, 1]],
                        [[0, 1, 1, 1], [1, 0, 1, 0]],
                        [[0, 1, 1, 1], [1, 0, 1, 1]],
                        [[0, 1, 1, 1], [1, 1, 0, 0]],
                        [[0, 1, 1, 1], [1, 1, 0, 1]],
                        [[0, 1, 1, 1], [1, 1, 1, 0]],
                        [[0, 1, 1, 1], [1, 1, 1, 1]],
                        [[1, 0, 0, 0], [0, 0, 0, 0]],
                        [[1, 0, 0, 0], [0, 0, 0, 1]],
                        [[1, 0, 0, 0], [0, 0, 1, 0]],
                        [[1, 0, 0, 0], [0, 0, 1, 1]],
                        [[1, 0, 0, 0], [0, 1, 0, 0]],
                        [[1, 0, 0, 0], [0, 1, 0, 1]],
                        [[1, 0, 0, 0], [0, 1, 1, 0]],
                        [[1, 0, 0, 0], [0, 1, 1, 1]],
                        [[1, 0, 0, 0], [1, 0, 0, 0]],
                        [[1, 0, 0, 0], [1, 0, 0, 1]],
                        [[1, 0, 0, 0], [1, 0, 1, 0]],
                        [[1, 0, 0, 0], [1, 0, 1, 1]],
                        [[1, 0, 0, 0], [1, 1, 0, 0]],
                        [[1, 0, 0, 0], [1, 1, 0, 1]],
                        [[1, 0, 0, 0], [1, 1, 1, 0]],
                        [[1, 0, 0, 0], [1, 1, 1, 1]],
                        [[1, 0, 0, 1], [0, 0, 0, 0]],
                        [[1, 0, 0, 1], [0, 0, 0, 1]],
                        [[1, 0, 0, 1], [0, 0, 1, 0]],
                        [[1, 0, 0, 1], [0, 0, 1, 1]],
                        [[1, 0, 0, 1], [0, 1, 0, 0]],
                        [[1, 0, 0, 1], [0, 1, 0, 1]],
                        [[1, 0, 0, 1], [0, 1, 1, 0]],
                        [[1, 0, 0, 1], [0, 1, 1, 1]],
                        [[1, 0, 0, 1], [1, 0, 0, 0]],
                        [[1, 0, 0, 1], [1, 0, 0, 1]],
                        [[1, 0, 0, 1], [1, 0, 1, 0]],
                        [[1, 0, 0, 1], [1, 0, 1, 1]],
                        [[1, 0, 0, 1], [1, 1, 0, 0]],
                        [[1, 0, 0, 1], [1, 1, 0, 1]],
                        [[1, 0, 0, 1], [1, 1, 1, 0]],
                        [[1, 0, 0, 1], [1, 1, 1, 1]],
                        [[1, 0, 1, 0], [0, 0, 0, 0]],
                        [[1, 0, 1, 0], [0, 0, 0, 1]],
                        [[1, 0, 1, 0], [0, 0, 1, 0]],
                        [[1, 0, 1, 0], [0, 0, 1, 1]],
                        [[1, 0, 1, 0], [0, 1, 0, 0]],
                        [[1, 0, 1, 0], [0, 1, 0, 1]],
                        [[1, 0, 1, 0], [0, 1, 1, 0]],
                        [[1, 0, 1, 0], [0, 1, 1, 1]],
                        [[1, 0, 1, 0], [1, 0, 0, 0]],
                        [[1, 0, 1, 0], [1, 0, 0, 1]],
                        [[1, 0, 1, 0], [1, 0, 1, 0]],
                        [[1, 0, 1, 0], [1, 0, 1, 1]],
                        [[1, 0, 1, 0], [1, 1, 0, 0]],
                        [[1, 0, 1, 0], [1, 1, 0, 1]],
                        [[1, 0, 1, 0], [1, 1, 1, 0]],
                        [[1, 0, 1, 0], [1, 1, 1, 1]],
                        [[1, 0, 1, 1], [0, 0, 0, 0]],
                        [[1, 0, 1, 1], [0, 0, 0, 1]],
                        [[1, 0, 1, 1], [0, 0, 1, 0]],
                        [[1, 0, 1, 1], [0, 0, 1, 1]],
                        [[1, 0, 1, 1], [0, 1, 0, 0]],
                        [[1, 0, 1, 1], [0, 1, 0, 1]],
                        [[1, 0, 1, 1], [0, 1, 1, 0]],
                        [[1, 0, 1, 1], [0, 1, 1, 1]],
                        [[1, 0, 1, 1], [1, 0, 0, 0]],
                        [[1, 0, 1, 1], [1, 0, 0, 1]],
                        [[1, 0, 1, 1], [1, 0, 1, 0]],
                        [[1, 0, 1, 1], [1, 0, 1, 1]],
                        [[1, 0, 1, 1], [1, 1, 0, 0]],
                        [[1, 0, 1, 1], [1, 1, 0, 1]],
                        [[1, 0, 1, 1], [1, 1, 1, 0]],
                        [[1, 0, 1, 1], [1, 1, 1, 1]],
                        [[1, 1, 0, 0], [0, 0, 0, 0]],
                        [[1, 1, 0, 0], [0, 0, 0, 1]],
                        [[1, 1, 0, 0], [0, 0, 1, 0]],
                        [[1, 1, 0, 0], [0, 0, 1, 1]],
                        [[1, 1, 0, 0], [0, 1, 0, 0]],
                        [[1, 1, 0, 0], [0, 1, 0, 1]],
                        [[1, 1, 0, 0], [0, 1, 1, 0]],
                        [[1, 1, 0, 0], [0, 1, 1, 1]],
                        [[1, 1, 0, 0], [1, 0, 0, 0]],
                        [[1, 1, 0, 0], [1, 0, 0, 1]],
                        [[1, 1, 0, 0], [1, 0, 1, 0]],
                        [[1, 1, 0, 0], [1, 0, 1, 1]],
                        [[1, 1, 0, 0], [1, 1, 0, 0]],
                        [[1, 1, 0, 0], [1, 1, 0, 1]],
                        [[1, 1, 0, 0], [1, 1, 1, 0]],
                        [[1, 1, 0, 0], [1, 1, 1, 1]],
                        [[1, 1, 0, 1], [0, 0, 0, 0]],
                        [[1, 1, 0, 1], [0, 0, 0, 1]],
                        [[1, 1, 0, 1], [0, 0, 1, 0]],
                        [[1, 1, 0, 1], [0, 0, 1, 1]],
                        [[1, 1, 0, 1], [0, 1, 0, 0]],
                        [[1, 1, 0, 1], [0, 1, 0, 1]],
                        [[1, 1, 0, 1], [0, 1, 1, 0]],
                        [[1, 1, 0, 1], [0, 1, 1, 1]],
                        [[1, 1, 0, 1], [1, 0, 0, 0]],
                        [[1, 1, 0, 1], [1, 0, 0, 1]],
                        [[1, 1, 0, 1], [1, 0, 1, 0]],
                        [[1, 1, 0, 1], [1, 0, 1, 1]],
                        [[1, 1, 0, 1], [1, 1, 0, 0]],
                        [[1, 1, 0, 1], [1, 1, 0, 1]],
                        [[1, 1, 0, 1], [1, 1, 1, 0]],
                        [[1, 1, 0, 1], [1, 1, 1, 1]],
                        [[1, 1, 1, 0], [0, 0, 0, 0]],
                        [[1, 1, 1, 0], [0, 0, 0, 1]],
                        [[1, 1, 1, 0], [0, 0, 1, 0]],
                        [[1, 1, 1, 0], [0, 0, 1, 1]],
                        [[1, 1, 1, 0], [0, 1, 0, 0]],
                        [[1, 1, 1, 0], [0, 1, 0, 1]],
                        [[1, 1, 1, 0], [0, 1, 1, 0]],
                        [[1, 1, 1, 0], [0, 1, 1, 1]],
                        [[1, 1, 1, 0], [1, 0, 0, 0]],
                        [[1, 1, 1, 0], [1, 0, 0, 1]],
                        [[1, 1, 1, 0], [1, 0, 1, 0]],
                        [[1, 1, 1, 0], [1, 0, 1, 1]],
                        [[1, 1, 1, 0], [1, 1, 0, 0]],
                        [[1, 1, 1, 0], [1, 1, 0, 1]],
                        [[1, 1, 1, 0], [1, 1, 1, 0]],
                        [[1, 1, 1, 0], [1, 1, 1, 1]],
                        [[1, 1, 1, 1], [0, 0, 0, 0]],
                        [[1, 1, 1, 1], [0, 0, 0, 1]],
                        [[1, 1, 1, 1], [0, 0, 1, 0]],
                        [[1, 1, 1, 1], [0, 0, 1, 1]],
                        [[1, 1, 1, 1], [0, 1, 0, 0]],
                        [[1, 1, 1, 1], [0, 1, 0, 1]],
                        [[1, 1, 1, 1], [0, 1, 1, 0]],
                        [[1, 1, 1, 1], [0, 1, 1, 1]],
                        [[1, 1, 1, 1], [1, 0, 0, 0]],
                        [[1, 1, 1, 1], [1, 0, 0, 1]],
                        [[1, 1, 1, 1], [1, 0, 1, 0]],
                        [[1, 1, 1, 1], [1, 0, 1, 1]],
                        [[1, 1, 1, 1], [1, 1, 0, 0]],
                        [[1, 1, 1, 1], [1, 1, 0, 1]],
                        [[1, 1, 1, 1], [1, 1, 1, 0]],
                        [[1, 1, 1, 1], [1, 1, 1, 1]] ], dtype = int)

Input_3D_array = np.array([ [0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0, 1, 1, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 1, 0, 0, 1],
                            [0, 0, 0, 0, 1, 0, 1, 0],
                            [0, 0, 0, 0, 1, 0, 1, 1],
                            [0, 0, 0, 0, 1, 1, 0, 0],
                            [0, 0, 0, 0, 1, 1, 0, 1],
                            [0, 0, 0, 0, 1, 1, 1, 0],
                            [0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 1, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 1, 0, 0, 1, 0],
                            [0, 0, 0, 1, 0, 0, 1, 1],
                            [0, 0, 0, 1, 0, 1, 0, 0],
                            [0, 0, 0, 1, 0, 1, 0, 1],
                            [0, 0, 0, 1, 0, 1, 1, 0],
                            [0, 0, 0, 1, 0, 1, 1, 1],
                            [0, 0, 0, 1, 1, 0, 0, 0],
                            [0, 0, 0, 1, 1, 0, 0, 1],
                            [0, 0, 0, 1, 1, 0, 1, 0],
                            [0, 0, 0, 1, 1, 0, 1, 1],
                            [0, 0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 1, 1, 0, 1],
                            [0, 0, 0, 1, 1, 1, 1, 0],
                            [0, 0, 0, 1, 1, 1, 1, 1],
                            [0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 1],
                            [0, 0, 1, 0, 0, 0, 1, 0],
                            [0, 0, 1, 0, 0, 0, 1, 1],
                            [0, 0, 1, 0, 0, 1, 0, 0],
                            [0, 0, 1, 0, 0, 1, 0, 1],
                            [0, 0, 1, 0, 0, 1, 1, 0],
                            [0, 0, 1, 0, 0, 1, 1, 1],
                            [0, 0, 1, 0, 1, 0, 0, 0],
                            [0, 0, 1, 0, 1, 0, 0, 1],
                            [0, 0, 1, 0, 1, 0, 1, 0],
                            [0, 0, 1, 0, 1, 0, 1, 1],
                            [0, 0, 1, 0, 1, 1, 0, 0],
                            [0, 0, 1, 0, 1, 1, 0, 1],
                            [0, 0, 1, 0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 1, 1, 1, 1],
                            [0, 0, 1, 1, 0, 0, 0, 0],
                            [0, 0, 1, 1, 0, 0, 0, 1],
                            [0, 0, 1, 1, 0, 0, 1, 0],
                            [0, 0, 1, 1, 0, 0, 1, 1],
                            [0, 0, 1, 1, 0, 1, 0, 0],
                            [0, 0, 1, 1, 0, 1, 0, 1],
                            [0, 0, 1, 1, 0, 1, 1, 0],
                            [0, 0, 1, 1, 0, 1, 1, 1],
                            [0, 0, 1, 1, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0, 1],
                            [0, 0, 1, 1, 1, 0, 1, 0],
                            [0, 0, 1, 1, 1, 0, 1, 1],
                            [0, 0, 1, 1, 1, 1, 0, 0],
                            [0, 0, 1, 1, 1, 1, 0, 1],
                            [0, 0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 1, 1, 1],
                            [0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0, 0, 1],
                            [0, 1, 0, 0, 0, 0, 1, 0],
                            [0, 1, 0, 0, 0, 0, 1, 1],
                            [0, 1, 0, 0, 0, 1, 0, 0],
                            [0, 1, 0, 0, 0, 1, 0, 1],
                            [0, 1, 0, 0, 0, 1, 1, 0],
                            [0, 1, 0, 0, 0, 1, 1, 1],
                            [0, 1, 0, 0, 1, 0, 0, 0],
                            [0, 1, 0, 0, 1, 0, 0, 1],
                            [0, 1, 0, 0, 1, 0, 1, 0],
                            [0, 1, 0, 0, 1, 0, 1, 1],
                            [0, 1, 0, 0, 1, 1, 0, 0],
                            [0, 1, 0, 0, 1, 1, 0, 1],
                            [0, 1, 0, 0, 1, 1, 1, 0],
                            [0, 1, 0, 0, 1, 1, 1, 1],
                            [0, 1, 0, 1, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 0, 1],
                            [0, 1, 0, 1, 0, 0, 1, 0],
                            [0, 1, 0, 1, 0, 0, 1, 1],
                            [0, 1, 0, 1, 0, 1, 0, 0],
                            [0, 1, 0, 1, 0, 1, 0, 1],
                            [0, 1, 0, 1, 0, 1, 1, 0],
                            [0, 1, 0, 1, 0, 1, 1, 1],
                            [0, 1, 0, 1, 1, 0, 0, 0],
                            [0, 1, 0, 1, 1, 0, 0, 1],
                            [0, 1, 0, 1, 1, 0, 1, 0],
                            [0, 1, 0, 1, 1, 0, 1, 1],
                            [0, 1, 0, 1, 1, 1, 0, 0],
                            [0, 1, 0, 1, 1, 1, 0, 1],
                            [0, 1, 0, 1, 1, 1, 1, 0],
                            [0, 1, 0, 1, 1, 1, 1, 1],
                            [0, 1, 1, 0, 0, 0, 0, 0],
                            [0, 1, 1, 0, 0, 0, 0, 1],
                            [0, 1, 1, 0, 0, 0, 1, 0],
                            [0, 1, 1, 0, 0, 0, 1, 1],
                            [0, 1, 1, 0, 0, 1, 0, 0],
                            [0, 1, 1, 0, 0, 1, 0, 1],
                            [0, 1, 1, 0, 0, 1, 1, 0],
                            [0, 1, 1, 0, 0, 1, 1, 1],
                            [0, 1, 1, 0, 1, 0, 0, 0],
                            [0, 1, 1, 0, 1, 0, 0, 1],
                            [0, 1, 1, 0, 1, 0, 1, 0],
                            [0, 1, 1, 0, 1, 0, 1, 1],
                            [0, 1, 1, 0, 1, 1, 0, 0],
                            [0, 1, 1, 0, 1, 1, 0, 1],
                            [0, 1, 1, 0, 1, 1, 1, 0],
                            [0, 1, 1, 0, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0, 0, 0, 0],
                            [0, 1, 1, 1, 0, 0, 0, 1],
                            [0, 1, 1, 1, 0, 0, 1, 0],
                            [0, 1, 1, 1, 0, 0, 1, 1],
                            [0, 1, 1, 1, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0, 1, 0, 1],
                            [0, 1, 1, 1, 0, 1, 1, 0],
                            [0, 1, 1, 1, 0, 1, 1, 1],
                            [0, 1, 1, 1, 1, 0, 0, 0],
                            [0, 1, 1, 1, 1, 0, 0, 1],
                            [0, 1, 1, 1, 1, 0, 1, 0],
                            [0, 1, 1, 1, 1, 0, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0, 1],
                            [0, 1, 1, 1, 1, 1, 1, 0],
                            [0, 1, 1, 1, 1, 1, 1, 1],
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 0, 1, 1],
                            [1, 0, 0, 0, 0, 1, 0, 0],
                            [1, 0, 0, 0, 0, 1, 0, 1],
                            [1, 0, 0, 0, 0, 1, 1, 0],
                            [1, 0, 0, 0, 0, 1, 1, 1],
                            [1, 0, 0, 0, 1, 0, 0, 0],
                            [1, 0, 0, 0, 1, 0, 0, 1],
                            [1, 0, 0, 0, 1, 0, 1, 0],
                            [1, 0, 0, 0, 1, 0, 1, 1],
                            [1, 0, 0, 0, 1, 1, 0, 0],
                            [1, 0, 0, 0, 1, 1, 0, 1],
                            [1, 0, 0, 0, 1, 1, 1, 0],
                            [1, 0, 0, 0, 1, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 0, 0],
                            [1, 0, 0, 1, 0, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 1, 0, 0, 1, 1],
                            [1, 0, 0, 1, 0, 1, 0, 0],
                            [1, 0, 0, 1, 0, 1, 0, 1],
                            [1, 0, 0, 1, 0, 1, 1, 0],
                            [1, 0, 0, 1, 0, 1, 1, 1],
                            [1, 0, 0, 1, 1, 0, 0, 0],
                            [1, 0, 0, 1, 1, 0, 0, 1],
                            [1, 0, 0, 1, 1, 0, 1, 0],
                            [1, 0, 0, 1, 1, 0, 1, 1],
                            [1, 0, 0, 1, 1, 1, 0, 0],
                            [1, 0, 0, 1, 1, 1, 0, 1],
                            [1, 0, 0, 1, 1, 1, 1, 0],
                            [1, 0, 0, 1, 1, 1, 1, 1],
                            [1, 0, 1, 0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 0, 0, 0, 1],
                            [1, 0, 1, 0, 0, 0, 1, 0],
                            [1, 0, 1, 0, 0, 0, 1, 1],
                            [1, 0, 1, 0, 0, 1, 0, 0],
                            [1, 0, 1, 0, 0, 1, 0, 1],
                            [1, 0, 1, 0, 0, 1, 1, 0],
                            [1, 0, 1, 0, 0, 1, 1, 1],
                            [1, 0, 1, 0, 1, 0, 0, 0],
                            [1, 0, 1, 0, 1, 0, 0, 1],
                            [1, 0, 1, 0, 1, 0, 1, 0],
                            [1, 0, 1, 0, 1, 0, 1, 1],
                            [1, 0, 1, 0, 1, 1, 0, 0],
                            [1, 0, 1, 0, 1, 1, 0, 1],
                            [1, 0, 1, 0, 1, 1, 1, 0],
                            [1, 0, 1, 0, 1, 1, 1, 1],
                            [1, 0, 1, 1, 0, 0, 0, 0],
                            [1, 0, 1, 1, 0, 0, 0, 1],
                            [1, 0, 1, 1, 0, 0, 1, 0],
                            [1, 0, 1, 1, 0, 0, 1, 1],
                            [1, 0, 1, 1, 0, 1, 0, 0],
                            [1, 0, 1, 1, 0, 1, 0, 1],
                            [1, 0, 1, 1, 0, 1, 1, 0],
                            [1, 0, 1, 1, 0, 1, 1, 1],
                            [1, 0, 1, 1, 1, 0, 0, 0],
                            [1, 0, 1, 1, 1, 0, 0, 1],
                            [1, 0, 1, 1, 1, 0, 1, 0],
                            [1, 0, 1, 1, 1, 0, 1, 1],
                            [1, 0, 1, 1, 1, 1, 0, 0],
                            [1, 0, 1, 1, 1, 1, 0, 1],
                            [1, 0, 1, 1, 1, 1, 1, 0],
                            [1, 0, 1, 1, 1, 1, 1, 1],
                            [1, 1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0, 1],
                            [1, 1, 0, 0, 0, 0, 1, 0],
                            [1, 1, 0, 0, 0, 0, 1, 1],
                            [1, 1, 0, 0, 0, 1, 0, 0],
                            [1, 1, 0, 0, 0, 1, 0, 1],
                            [1, 1, 0, 0, 0, 1, 1, 0],
                            [1, 1, 0, 0, 0, 1, 1, 1],
                            [1, 1, 0, 0, 1, 0, 0, 0],
                            [1, 1, 0, 0, 1, 0, 0, 1],
                            [1, 1, 0, 0, 1, 0, 1, 0],
                            [1, 1, 0, 0, 1, 0, 1, 1],
                            [1, 1, 0, 0, 1, 1, 0, 0],
                            [1, 1, 0, 0, 1, 1, 0, 1],
                            [1, 1, 0, 0, 1, 1, 1, 0],
                            [1, 1, 0, 0, 1, 1, 1, 1],
                            [1, 1, 0, 1, 0, 0, 0, 0],
                            [1, 1, 0, 1, 0, 0, 0, 1],
                            [1, 1, 0, 1, 0, 0, 1, 0],
                            [1, 1, 0, 1, 0, 0, 1, 1],
                            [1, 1, 0, 1, 0, 1, 0, 0],
                            [1, 1, 0, 1, 0, 1, 0, 1],
                            [1, 1, 0, 1, 0, 1, 1, 0],
                            [1, 1, 0, 1, 0, 1, 1, 1],
                            [1, 1, 0, 1, 1, 0, 0, 0],
                            [1, 1, 0, 1, 1, 0, 0, 1],
                            [1, 1, 0, 1, 1, 0, 1, 0],
                            [1, 1, 0, 1, 1, 0, 1, 1],
                            [1, 1, 0, 1, 1, 1, 0, 0],
                            [1, 1, 0, 1, 1, 1, 0, 1],
                            [1, 1, 0, 1, 1, 1, 1, 0],
                            [1, 1, 0, 1, 1, 1, 1, 1],
                            [1, 1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0, 1],
                            [1, 1, 1, 0, 0, 0, 1, 0],
                            [1, 1, 1, 0, 0, 0, 1, 1],
                            [1, 1, 1, 0, 0, 1, 0, 0],
                            [1, 1, 1, 0, 0, 1, 0, 1],
                            [1, 1, 1, 0, 0, 1, 1, 0],
                            [1, 1, 1, 0, 0, 1, 1, 1],
                            [1, 1, 1, 0, 1, 0, 0, 0],
                            [1, 1, 1, 0, 1, 0, 0, 1],
                            [1, 1, 1, 0, 1, 0, 1, 0],
                            [1, 1, 1, 0, 1, 0, 1, 1],
                            [1, 1, 1, 0, 1, 1, 0, 0],
                            [1, 1, 1, 0, 1, 1, 0, 1],
                            [1, 1, 1, 0, 1, 1, 1, 0],
                            [1, 1, 1, 0, 1, 1, 1, 1],
                            [1, 1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0, 1],
                            [1, 1, 1, 1, 0, 0, 1, 0],
                            [1, 1, 1, 1, 0, 0, 1, 1],
                            [1, 1, 1, 1, 0, 1, 0, 0],
                            [1, 1, 1, 1, 0, 1, 0, 1],
                            [1, 1, 1, 1, 0, 1, 1, 0],
                            [1, 1, 1, 1, 0, 1, 1, 1],
                            [1, 1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 1, 1, 0, 0, 1],
                            [1, 1, 1, 1, 1, 0, 1, 0],
                            [1, 1, 1, 1, 1, 0, 1, 1],
                            [1, 1, 1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1, 0, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1, 1] ], dtype = int)

Output_3D = np.array([  0, 0, 1, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -2, -1, -2, -1, -1, -1, -1,
                        0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, -1, 0, -1, 0, 0, 0, 0,
                        0, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = int)

Output_3D_array = np.array([  0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 2, 2, 2, 2, 2, 2, 3, 2, 3, 2, 2, 2, 2,
                        0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0,
                        0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 2, 0, 2, 0, 0, 0, 0,
                        0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype = int)