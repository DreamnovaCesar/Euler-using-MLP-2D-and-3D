
import argparse

from Article_Euler_Number_3D_General import *
from Article_Euler_Number_2D_And_3D_ML import *

from Article_Euler_Number_Create_Data import DataEuler

def parse_opt():

    # * Construct the argument parser
    parser = argparse.ArgumentParser()

    # * Add the arguments to the parser
    parser.add_argument("--C2D", type = bool, help = "Bool 2D")
    parser.add_argument("--save-folder2D", default = r'C:\Users\Cesar\Desktop\Codes\Python\Article_Euler_2D_and_3D\ObjectsArg\2D', 
                        type = str, help = "Folder")
    parser.add_argument("--NO2D", type = int, help = "Number of objects 2D")
    parser.add_argument("--H2D", type = int, help = "Height 2D")
    parser.add_argument("--W2D", type = int, help = "Width 2D")
    
    parser.add_argument("--C3D", type = bool, help = "Bool 3D")
    parser.add_argument("--save-folder3D", default = r'C:\Users\Cesar\Desktop\Codes\Python\Article_Euler_2D_and_3D\ObjectsArg\3D', 
                        type = str, help = "Folder")
    parser.add_argument("--NO3D", type = int, help = "Number of objects 3D")
    parser.add_argument("--H3D", type = int, help = "Height 3D")
    parser.add_argument("--W3D", type = int, help = "Width 3D")
    parser.add_argument("--D3D", type = int, help = "Width 3D")
    opt = parser.parse_args()

    #print("{}, {}, {}".format(opt.NO2D, opt.H2D, opt.W2D))
    #print("{}, {}, {}, {}, {}".format(opt.save_folder3D, opt.NO3D, opt.H3D, opt.W3D, opt.D3D))

    if(opt.C2D):

        Images_2D = DataEuler(folder = opt.save_folder2D, NI = opt.NO2D, Height = opt.H2D, Width = opt.W2D);

        Images_2D.create_data_euler_2D_random();

    if(opt.C3D):

        Images_3D = DataEuler(folder = opt.save_folder3D, NI = opt.NO3D, Height = opt.H3D, Width = opt.W3D, Depth = opt.D3D);

        Images_3D.create_data_euler_3D_random();

# ?
def main():

    parse_opt()

# ?
if __name__ == "__main__":
    main()