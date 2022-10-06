from typing import Any

import numpy as np

from diagrams import Diagram
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.aws.network import ELB

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

# ?
def project_diagram() -> None:

    with Diagram("Project diagram", show = False):
        ELB("lb") >> EC2("web") >> RDS("userdb")