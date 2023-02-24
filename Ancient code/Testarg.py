import argparse
from Article_Euler_Number_Class_EulerImageGenerator import DataEuler

# Construct the argument parser
parser = argparse.ArgumentParser()

# Add the arguments to the parser
parser.add_argument("--NO", type = int, help = "Number of objects 2D")
parser.add_argument("--H", type = int, help = "Height")
parser.add_argument("--W", type = int, help = "Width")
opt = parser.parse_args()

# Calculate the sum
print("{}, {}, {}".format(opt.NO, opt.H, opt.W))

Images_2D = DataEuler(folder = r'C:\Users\Cesar\Desktop\Codes\Python\Article_Euler_2D_and_3D\ObjectsArg', 
                        NI = opt.NO, Height = opt.H, Width = opt.W);

Images_2D.create_data_euler_2D_random();
