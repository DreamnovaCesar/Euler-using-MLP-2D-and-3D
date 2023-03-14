import numpy as np

# ?                     # a, b, c, d    e, f, g, h
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

# ?
_INPUT_3D_ = np.array([ [0, 0, 0, 0, 0, 0, 0, 0],
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
_OUTPUT_3D_ = np.array([  0, 0, 1, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
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
