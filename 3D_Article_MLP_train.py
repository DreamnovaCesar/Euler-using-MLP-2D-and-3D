import numpy as np

from General_3D import Input_3D
from General_3D import Input_3D_array
from General_3D import Output_3D
from General_3D import Output_3D_array

from MLP_2D_3D import model_euler_3D
from MLP_2D_3D import plot_data

print(Input_3D_array.shape)
print(Output_3D_array.shape)

Hist_data = model_euler_3D(Input_3D_array, Output_3D_array, 2000, 'Model_3D')
plot_data(Hist_data)
