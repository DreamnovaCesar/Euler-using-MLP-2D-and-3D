
import numpy as np

from S_3D_Article_General import Input_3D_array
from S_3D_Article_General import Output_3D_array
from S_3D_2D_Article_ML_Functions import EulerNumberML


def test():
    
    Euler_path_3D = r'C:\Users\Cesar\Dropbox\PC\Desktop\Euler 3D'
    Object_path = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_3D_1.txt"

    #Euler_3D_MLP = EulerNumberML(input = Input_3D_array, output = Output_3D_array, object = Object_path, folder = Euler_path_3D, modelname = 'MLP_3D', epochs = 100)
    Euler_3D_RF = EulerNumberML(input = Input_3D_array, output = Output_3D_array, object = Object_path, folder = Euler_path_3D, modelname = 'RF_3D', epochs = 100)

    #Euler_3D_MLP.print_octovoxel_order()
    Euler_3D_RF.print_octovoxel_order()

    #Euler_3D_MLP.model_euler_3D_MLP()
    Euler_3D_RF.model_euler_3D_RF()

    #Array_MLP = Euler_3D_MLP.obtain_arrays_from_object()
    Array_RF = Euler_3D_RF.obtain_arrays_from_object()

    #Euler_3D_MLP.model_prediction_3D('MLP_3D.h5', Array_MLP)
    Euler_3D_RF.model_prediction_3D('RF_3D.joblib', Array_RF)
    
def main():

    Tuple_index = (2, 9, 11, 24, 25, 26, 27, 33, 35, 36, 37, 38, 39, 
                    40, 41, 42, 43, 44, 45, 46, 47, 56, 57, 58, 59, 129, 
                        131, 137, 139, 148, 149, 150, 151, 156, 157, 158, 
                            159, 161, 163, 169, 171, 180, 181, 182, 183, 188, 
                                189, 190, 191)

    #Tuple_value = (1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
    #               -1, -2, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, 
    #                   -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, 
    #                       -1, 1, 1, 1, 1, 1, 1, 1, 1)
    
    Tuple_value = (1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                    2, 3, 2, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 
                        2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 
                            2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1)


if __name__ == "__main__":
    test()