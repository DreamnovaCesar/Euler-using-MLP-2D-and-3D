from Article_Euler_Number_2D_General import Input_2D
from Article_Euler_Number_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_3D_General import Input_3D_array
from Article_Euler_Number_3D_General import Output_3D_array

from Article_Euler_Number_3D_General import *

from Article_Euler_Number_Create_Data import DataEuler
from Article_Euler_Menu_Console import Menu
from Article_Euler_Menu_Tkinter import MenuTkinter

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

    Euler_path_2D = r'Objects\Handcraft\3D\Data'
    Object_path_1 = r"Objects\Handcraft\2D\Example_2D_1.txt"
    Object_path_2 = r"Objects\Handcraft\2D\Example_2D_2.txt"
    Object_path_3 = r"Objects\Handcraft\2D\Example_2D_3.txt"
    Object_path_4 = r"Objects\Handcraft\2D\Example_2D_4.txt"

    Euler_2D_MLP_4 = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D, MN = 'Model_MLP_2D_4', epochs = 2000)
    Euler_2D_MLP_8 = EulerNumberML2D(input = Input_2D, output = Output_2D_8_Connectivity, folder = Euler_path_2D, MN = 'Model_MLP_2D_8', epochs = 2000)

    Euler_2D_MLP_4.model_euler_MLP_2D()
    Euler_2D_MLP_8.model_euler_MLP_2D()

    Array_MLP_4 = Euler_2D_MLP_4.obtain_arrays_from_object_2D(Object_path_4)
    Array_MLP_8 = Euler_2D_MLP_8.obtain_arrays_from_object_2D(Object_path_4)

    Euler_2D_MLP_4.model_prediction_2D('Model_MLP_2D_4.h5', Array_MLP_4)
    Euler_2D_MLP_8.model_prediction_2D('Model_MLP_2D_8.h5', Array_MLP_8)

    Euler_2D_MLP_4.connectivity_4_prediction_2D(Array_MLP_4)
    Euler_2D_MLP_8.connectivity_8_prediction_2D(Array_MLP_8)

def Euler_3D_test_handcraft():
    
    global Input_3D_array
    global Output_3D_array

    Euler_path_3D = r'Objects\Handcraft\3D\Data'
    Object_path_1 = r"Objects\Handcraft\3D\Example_3D_1.txt"
    Object_path_2 = r"Objects\Handcraft\3D\Example_3D_2.txt"
    Object_path_3 = r"Objects\Handcraft\3D\Example_3D_3.txt"
    Object_path_4 = r"Objects\Handcraft\3D\Example_3D_4.txt"

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

    Euler_path_2D= r'Objects\2D'
    Euler_path_2D_data = r'Objects\2D\Data'
    Object_1_2D = r"Objects\2D\Image_2D_0.txt"


    Euler_2D_MLP = EulerNumberML2D(input = Input_2D, output = Output_2D_4_Connectivity, folder = Euler_path_2D, MN = 'Model_MLP_2D_4', epochs = 1000)

    Euler_2D_MLP.model_euler_MLP_2D()

    Array_MLP = Euler_2D_MLP.obtain_arrays_from_object_2D(Object_1_2D)

    Euler_2D_MLP.model_prediction_2D('Model_MLP_2D_4.h5', Array_MLP)

    Euler_2D_MLP.connectivity_4_prediction_2D(Array_MLP)
    Euler_2D_MLP.connectivity_8_prediction_2D(Array_MLP)

def Euler_3D_test():
    
    global Input_3D_array
    global Output_3D_array

    Euler_path_3D = r'Objects\3D'
    Euler_path_3D_data = r'Objects\3D\Data'
    Object_1_3D = r"Objects\3D\Image_3D_0.txt"


    #Euler_3D_MLP = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D, modelname = 'Model_MLP_3D', epochs = 100)
    Euler_3D_RF = EulerNumberML3D(input = Input_3D_array, output = Output_3D_array, folder = Euler_path_3D, MN = 'Model_RF_3D', epochs = 100)

    #Euler_3D_MLP.print_octovoxel_order_3D()
    #Euler_3D_RF.print_octovoxel_order_3D()

    #Euler_3D_MLP.model_euler_MLP_3D()
    Euler_3D_RF.model_euler_RF_3D()

    #Array_MLP = Euler_3D_MLP.obtain_arrays_from_object_3D(Object_path_4)
    Array_RF = Euler_3D_RF.obtain_arrays_from_object_3D(Object_1_3D)

    #Euler_3D_MLP.model_prediction_3D('Model_MLP_3D.h5', Array_MLP)
    Euler_3D_RF.model_prediction_3D('Model_RF_3D.joblib', Array_RF)
    Euler_3D_RF.Show_array_3D(Object_1_3D)

def Create_objects():

    global Input_3D_array
    global Output_3D_array
    
    Folder_2D = r'Objects\2D\Images';

    Images_2D = DataEuler(folder = Folder_2D, NI = 10, Height = 8, Width = 8);
    Images_2D.create_data_euler_2D_random();

def read_image_with_metadata_2D(Array_file: str) -> np.ndarray:
        """
        Static method to load txt and convert it into tensors

        Args:
            Array_file (str): description
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

def obtain_arrays_from_object_2D(Object) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 2D array

        Args:
            Object (str): description

        """

        # *
        Arrays = []
        Asterisks = 30

        k = 2

        # *
        Array = read_image_with_metadata_2D(Object)
        Qs = []
        
        # *
        Q1 = np.array([ [ 0,  0],
                        [ 0,  0]    ], dtype = 'int')

        Q2 = np.array([ [ 0,  0],
                        [ 0,  1]    ], dtype = 'int')
        
        Q3 = np.array([ [ 0,  0],
                        [ 1,  0]    ], dtype = 'int')

        Q4 = np.array([ [ 0,  0],
                        [ 1,  1]    ], dtype = 'int')

        Q5 = np.array([ [ 0,  1],
                        [ 0,  0]    ], dtype = 'int')

        Q6 = np.array([ [ 0,  1],
                        [ 0,  1]    ], dtype = 'int')

        Q7 = np.array([ [ 0,  1],
                        [ 1,  0]    ], dtype = 'int')

        Q8 = np.array([ [ 0,  1],
                        [ 1,  1]    ], dtype = 'int')

        Q9 = np.array([ [ 1,  0],
                        [ 0,  0]    ], dtype = 'int')

        Q10 = np.array([ [ 1,  0],
                         [ 0,  1]    ], dtype = 'int')

        Q11 = np.array([    [ 1,  0],
                            [ 1,  0]    ], dtype = 'int')

        Q12 = np.array([    [ 1,  0],
                            [ 1,  1]    ], dtype = 'int')
        
        Q13 = np.array([    [ 1,  1],
                            [ 0,  0]    ], dtype = 'int')
        
        Q14 = np.array([    [ 1,  1],
                            [ 0,  1]    ], dtype = 'int')
        
        Q15 = np.array([    [ 1,  1],
                            [ 1,  0]    ], dtype = 'int')
        
        Q16 = np.array([    [ 1,  1],
                            [ 1,  1]    ], dtype = 'int')
        
        Qs.extend((Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9, Q10, Q11, Q12, Q13, Q14, Q15, Q16))
        Qs_value = np.zeros((16), dtype = 'int')

        Array_comparison = np.zeros((k, k), dtype = 'int')
        Array_prediction = np.zeros((4), dtype = 'int')

        for i in range(Array.shape[0] - 1):
            for j in range(Array.shape[1] - 1):
                
                #Array_comparison[0][0] = Array[i][j]
                #Array_comparison[1][0] = Array[i + 1][j]
                #Array_comparison[0][1] = Array[i][j + 1]
                #Array_comparison[1][1] = Array[i + 1][j + 1]

                #Array_prediction[0] = Array[i][j]
                #Array_prediction[1] = Array[i][j + 1]
                #Array_prediction[2] = Array[i + 1][j]
                #Array_prediction[3] = Array[i + 1][j + 1]

                for Index in range(len(Qs)):
                    
                    print('Kernel: {}'.format(Array[i:k + i, j:k + j]))
                    print('Qs: {}'.format(Qs[Index]))
                    print('\n')
                    print('\n')

                    if(np.array_equal(Array[i:k + i, j:k + j], Qs[Index])):
                        Qs_value[Index] += 1
                        print('Q{}_value: {}'.format(Index, Qs_value[Index]))
                
                print(Qs_value)
                print('\n')
                #print("*" * Asterisks)

                # *
                print("*" * Asterisks)
                #Array_prediction_list = Array_prediction.tolist()
                #Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                # *
                #print("Kernel array")
                #print(Array[i:k + i, j:k + j])
                #print('\n')
                #print("Prediction array")
                #print(Array_prediction)
                #print('\n')
                #Arrays.append(Array_prediction_list_int)
                print("*" * Asterisks)
                print('\n')

        # *
        #for i in range(len(Arrays)):
            #print('{} ---- {}'.format(i, Arrays[i]))
        #print('\n')
        
        return Arrays

def read_image_with_metadata_3D(Array_file: str) -> np.ndarray:
        """
        Static method to load txt and convert it into tensors

        Args:
            Array_file (str): description
        """
        # *
        Array = np.loadtxt(Array_file, delimiter = ',')
        
        # *
        Height = Array.shape[0]/Array.shape[1]
        Array_new = Array.reshape(int(Height), int(Array.shape[1]), int(Array.shape[1]))

        # *
        Array_new = Array_new.astype(int)

        print('\n')
        print('Array obtained')
        print('\n')
        print(Array_new)
        print('\n')
        print('Number of channels: {}'.format(Array_new.shape[0]))
        print('\n')
        print('Number of rows: {}'.format(Array_new.shape[1]))
        print('\n')
        print('Number of columns: {}'.format(Array_new.shape[2]))
        print('\n')

        return Array_new

# ?
def main():
    """Main function
    """
    #Create_objects()

    #obtain_arrays_from_object_2D(r'Objects\2D\Images\Image_random_0_2D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_0_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_1_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_2_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_3_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_4_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_5_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_6_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_12_3D.txt')
    #obtain_arrays_from_object_3D(r'Objects\3D\Images\Image_random_16_3D.txt')

    config = MenuTkinter()
    config.menu()

# ?
if __name__ == "__main__":
    main()