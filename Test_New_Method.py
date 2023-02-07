from New_Method_Euler_Number_3D import OctovoxelEulerANN

OCANN = OctovoxelEulerANN(folder = r'Objects\3D\Data', epochs = 3000, MN = 'Test_1')
OCANN.MLP_octovoxel_training_3D(r'Objects\3D\Data\Dataframe_test.csv', "ADAGRAD", 0.001)
#OCANN.MLP_octovoxel_prediction_3D(r'Objects\3D\Data\Test_1.h5', r'Objects\3D\Images backup\Image_random_3_3D.txt')