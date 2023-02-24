from Article_Euler_Number_Info_2D_General import Input_2D
from Article_Euler_Number_Info_2D_General import Output_2D_4_Connectivity
from Article_Euler_Number_Info_2D_General import Output_2D_8_Connectivity

from Article_Euler_Number_Info_3D_General import Input_3D_array
from Article_Euler_Number_Info_3D_General import Output_3D_array

from Article_Euler_Number_Info_3D_General import *
from Article_Euler_Number_Libraries import *

#from Article_Euler_Menu_Console import Menu
#from Article_Euler_Menu_Tkinter import MenuTkinter

from Article_Euler_Number_Class_EulerExtractorUtilities import EulerExtractorUtilities
from Article_Euler_Number_Class_EulerExtractorUtilities import EulerExtractor2D
from Article_Euler_Number_Class_EulerExtractorUtilities import EulerExtractor3D


def Euler_3D():
    
    Euler_path_3D = r'Objects\3D\Data'

    Euler = EulerExtractor3D(Folder = Euler_path_3D, ModelName = 'Model_MLP_Octo_3D')

    #Euler.MLP_training_octovoxels(r'Objects\3D\Data\Dataframe_test.csv', "Adagrad", 0.001, 500)
    #Euler.get_number_octovoxels(r'Objects\3D\Images backup\Image_random_3_3D.txt')
    Euler.MLP_predict_octovoxels(r'Objects\3D\Data\Model_MLP_Octo_3D.h5', r'Objects\3D\Images backup\Image_random_3_3D.txt')


# ?
def main():
    """Main function
    """
    Euler_3D()

    #config = MenuTkinter()
    #config.menu()

# ?
if __name__ == "__main__":
    main()