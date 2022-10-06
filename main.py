
import numpy as np

from Article_Euler_Number_2D_General import Input_2D
from Article_Euler_Number_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_3D_General import project_diagram
from Article_Euler_Number_3D_General import Input_3D_array
from Article_Euler_Number_3D_General import Output_3D_array

from Article_Euler_Number_2D_And_3D_ML import *


def Variables():

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

def Euler_2D_test():
    
    Euler_path_2D = r'C:\Users\Cesar\Dropbox\PC\Desktop\Euler 3D'
    Object_path_1 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_1.txt"
    Object_path_2 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_2.txt"
    Object_path_3 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_3.txt"
    Object_path_4 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_4.txt"

    Euler_2D_MLP = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D, modelname = 'Model_MLP_2D_4', epochs = 1000)

    Euler_2D_MLP.model_euler_2D_MLP()

    Array_MLP = Euler_2D_MLP.obtain_arrays_from_object_2D(Object_path_4)

    #Euler_2D_MLP.model_prediction_2D('Model_MLP_2D_4.h5', Array_MLP)

    #Euler_2D_MLP.connectivity_4_prediction_2D(Array_MLP)
    Euler_2D_MLP.connectivity_8_prediction_2D(Array_MLP)


def Euler_3D_test():
    
    Euler_path_3D = r'C:\Users\Cesar\Dropbox\PC\Desktop\Euler 3D'
    Object_path_1 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_1.txt"
    Object_path_2 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_2.txt"
    Object_path_3 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_3.txt"
    Object_path_4 = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Euler 3D\Example_3D_4.txt"

    #Euler_3D_MLP = EulerNumberML(input = Input_3D_array, output = Output_3D_array, object = Object_path, folder = Euler_path_3D, modelname = 'Model_MLP_3D', epochs = 100)
    Euler_3D_RF = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D, modelname = 'Model_RF_3D', epochs = 100)

    #Euler_3D_MLP.print_octovoxel_order()
    Euler_3D_RF.print_octovoxel_order_3D()

    #Euler_3D_MLP.model_euler_3D_MLP()
    Euler_3D_RF.model_euler_3D_RF()

    #Array_MLP = Euler_3D_MLP.obtain_arrays_from_object()
    Array_RF = Euler_3D_RF.obtain_arrays_from_object_3D(Object_path_4)

    #Euler_3D_MLP.model_prediction_3D('MLP_3D.h5', Array_MLP)
    Euler_3D_RF.model_prediction_3D('Model_RF_3D.joblib', Array_RF)
    
def main():
    #project_diagram()
    #Euler_2D_test()
    Euler_3D_test()


if __name__ == "__main__":
    main()