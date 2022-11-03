
from Article_Euler_Number_3D_General import project_diagram

from Article_Euler_Number_2D_General import Input_2D
from Article_Euler_Number_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_3D_General import Input_3D_array
from Article_Euler_Number_3D_General import Output_3D_array

from Article_Euler_Number_Create_Data import DataEuler

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

def Euler_2D_test_handcraft():
    
    global Input_2D
    global Output_2D_4_Connectivity
    global Output_2D_8_Connectivity

    Euler_path_2D = r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\2D\Data'
    Object_path_1 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\2D\Example_2D_1.txt"
    Object_path_2 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\2D\Example_2D_1.txt"
    Object_path_3 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\2D\Example_2D_1.txt"
    Object_path_4 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\2D\Example_2D_1.txt"

    Euler_2D_MLP_4 = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D, MN = 'Model_MLP_2D_4', epochs = 1000)
    Euler_2D_MLP_8 = EulerNumberML2D(input = Input_2D, output = Output_2D_8_Connectivity, folder = Euler_path_2D, MN = 'Model_MLP_2D_8', epochs = 1000)

    Euler_2D_MLP_4.model_euler_MLP_2D()
    Euler_2D_MLP_8.model_euler_MLP_2D()

    Array_MLP_4 = Euler_2D_MLP_4.obtain_arrays_from_object_2D(Object_path_4)
    Array_MLP_8 = Euler_2D_MLP_4.obtain_arrays_from_object_2D(Object_path_4)

    Euler_2D_MLP_4.model_prediction_2D('Model_MLP_2D_4.h5', Array_MLP_4)
    Euler_2D_MLP_8.model_prediction_2D('Model_MLP_2D_8.h5', Array_MLP_8)

    #Euler_2D_MLP_4.connectivity_4_prediction_2D(Array_MLP_4)
    #Euler_2D_MLP_8.connectivity_8_prediction_2D(Array_MLP_8)


def Euler_3D_test_handcraft():
    
    global Input_3D_array
    global Output_3D_array

    Euler_path_3D = r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\3D\Data'
    Object_path_1 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\3D\Example_3D_1.txt"
    Object_path_2 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\3D\Example_3D_2.txt"
    Object_path_3 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\3D\Example_3D_3.txt"
    Object_path_4 = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\3D\Example_3D_4.txt"

    #Euler_3D_MLP = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D, modelname = 'Model_MLP_3D', epochs = 100)
    Euler_3D_RF = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D, MN = 'Model_RF_3D', epochs = 100)

    #Euler_3D_MLP.print_octovoxel_order_3D()
    #Euler_3D_RF.print_octovoxel_order_3D()

    #Euler_3D_MLP.model_euler_MLP_3D()
    #Euler_3D_RF.model_euler_RF_3D()

    #Array_MLP = Euler_3D_MLP.obtain_arrays_from_object_3D(Object_path_4)
    Array_RF = Euler_3D_RF.obtain_arrays_from_object_3D(Object_path_2)

    #Euler_3D_MLP.model_prediction_3D('Model_MLP_3D.h5', Array_MLP)
    Euler_3D_RF.model_prediction_3D('Model_RF_3D.joblib', Array_RF)
    Euler_3D_RF.Show_array_3D(Object_path_2)

def Euler_2D_test():
    
    global Input_2D
    global Output_2D_4_Connectivity
    global Output_2D_8_Connectivity

    Euler_path_2D= r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\2D'
    Euler_path_2D_data = r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\2D\Data'
    Object_1_2D = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\2D\Image_2D_0.txt"


    Euler_2D_MLP = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D, MN = 'Model_MLP_2D_4', epochs = 1000)

    Euler_2D_MLP.model_euler_MLP_2D()

    Array_MLP = Euler_2D_MLP.obtain_arrays_from_object_2D(Object_1_2D)

    Euler_2D_MLP.model_prediction_2D('Model_MLP_2D_4.h5', Array_MLP)

    Euler_2D_MLP.connectivity_4_prediction_2D(Array_MLP)
    Euler_2D_MLP.connectivity_8_prediction_2D(Array_MLP)


def Euler_3D_test():
    
    global Input_3D_array
    global Output_3D_array

    Euler_path_3D = r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\Handcraft\3D'
    Euler_path_3D_data = r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\3D\Data'
    Object_1_3D = r"C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\3D\Image_3D_0.txt"


    #Euler_3D_MLP = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D, modelname = 'Model_MLP_3D', epochs = 100)
    Euler_3D_RF = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D, MN = 'Model_RF_3D', epochs = 100)

    #Euler_3D_MLP.print_octovoxel_order_3D()
    #Euler_3D_RF.print_octovoxel_order_3D()

    #Euler_3D_MLP.model_euler_MLP_3D()
    Euler_3D_RF.model_euler_RF_3D()

    #Array_MLP = Euler_3D_MLP.obtain_arrays_from_object_3D(Object_path_4)
    #Array_RF = Euler_3D_RF.obtain_arrays_from_object_3D(Object_1_3D)

    #Euler_3D_MLP.model_prediction_3D('Model_MLP_3D.h5', Array_MLP)
    #Euler_3D_RF.model_prediction_3D('Model_RF_3D.joblib', Array_RF)

def Create_objects():

    Folder_2D = r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\2D';
    Folder_3D = r'C:\Users\Cesar\Desktop\Python software\Dr.Hermilo 3D\Objects\3D';

    Images_2D = DataEuler(folder = Folder_2D, NI = 10, Height = 8, Width = 8, EN = 2, MT = 'Model_MLP_2D_4.h5');
    Images_2D.create_data_euler_2D_settings();

    #Images_3D = DataEuler(folder = Folder_3D, NI = 10, Height = 8, Width = 8, Depth = 8);
    #Images_3D.create_data_euler_3D_settings();

def main():

    #Create_objects()
    Euler_2D_test_handcraft()
    #Euler_3D_test_handcraft()

    #Create_objects()
    
    #Euler_3D_test()
    #Euler_3D_test()


if __name__ == "__main__":
    main()