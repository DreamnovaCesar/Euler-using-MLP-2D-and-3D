import numpy as np

from S_3D_Article_General import Input_3D
from S_3D_Article_General import Input_3D_array
from S_3D_Article_General import Output_3D
from S_3D_Article_General import Output_3D_array

from S_3D_2D_Article_ML_Functions import model_euler_3D
from S_3D_2D_Article_ML_Functions import plot_data_accuracy
from S_3D_2D_Article_ML_Functions import plot_data_loss

Hist_data = model_euler_3D(Input_3D_array, Output_3D_array, 500, 'Model_3D')

plot_data_accuracy(Hist_data)
plot_data_loss(Hist_data)
