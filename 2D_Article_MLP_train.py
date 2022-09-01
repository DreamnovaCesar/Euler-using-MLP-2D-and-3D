import numpy as np

from General_2D import Input_2D
from General_2D import Output_2D_4_Connectivity
from General_2D import Output_2D_8_Connectivity

from MLP_2D_3D import model_euler_2D
from MLP_2D_3D import plot_data

Hist_data = model_euler_2D(Input_2D, Output_2D_4_Connectivity, 2000, 'Model_4_connection')
plot_data(Hist_data)

Hist_data = model_euler_2D(Input_2D, Output_2D_8_Connectivity, 2000, 'Model_8_connection')
plot_data(Hist_data)