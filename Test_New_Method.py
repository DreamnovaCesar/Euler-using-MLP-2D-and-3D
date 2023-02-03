from New_Method_Euler_Number_3D import OctovoxelEulerANN

OCANN = OctovoxelEulerANN(folder = r'Objects\3D\Data', epochs = 100, MN = 'Test_1')
OCANN.MLP_octovoxel_training_3D(r'Objects\3D\Data\Dataframe_test.csv')