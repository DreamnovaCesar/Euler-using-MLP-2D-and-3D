import tensorflow as tf
import pandas as pd
import numpy as np
from Article_Euler_Number_Info_3D_General import *

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

def get_octovoxel_3D(Object: str) -> list[np.ndarray]:
        """
        Method to obtain 1D arrays from a 3D array

        Args:
            Object (str): description

        """

        #Array = np.loadtxt(self.Object, delimiter = ',')

        # *
        Arrays = []
        Asterisks = 30

        l = 2

        # *
        Array_new = read_image_with_metadata_3D(Object)

        # * Creation of empty numpy arrays 3D

        Qs = table_binary_multi_256(256)
        Qs_value = np.zeros((256), dtype = 'int')

        # *
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    # *
                    #Array_prediction_octov[0][0][0] = Array_new[i][j][k]
                    #Array_prediction_octov[0][0][1] = Array_new[i][j][k + 1]

                    #Array_prediction_octov[0][1][0] = Array_new[i][j + 1][k]
                    #Array_prediction_octov[0][1][1] = Array_new[i][j + 1][k + 1]

                    #Array_prediction_octov[1][0][0] = Array_new[i + 1][j][k]
                    #Array_prediction_octov[1][0][1] = Array_new[i + 1][j][k + 1]

                    #Array_prediction_octov[1][1][0] = Array_new[i + 1][j + 1][k]
                    #Array_prediction_octov[1][1][1] = Array_new[i + 1][j + 1][k + 1]

                    #Array_new[i:l + i, j:l + j, k:l + k]

                    # *
                    #Array_prediction[0] = Array_new[i + 1][j][k]
                    #Array_prediction[1] = Array_new[i + 1][j][k + 1]

                    #Array_prediction[2] = Array_new[i][j][k]
                    #Array_prediction[3] = Array_new[i][j][k + 1]

                    #Array_prediction[4] = Array_new[i + 1][j + 1][k]
                    #Array_prediction[5] = Array_new[i + 1][j + 1][k + 1]

                    #Array_prediction[6] = Array_new[i][j + 1][k]
                    #Array_prediction[7] = Array_new[i][j + 1][k + 1]
                    #print('\n')

                    for Index in range(len(Qs)):
                    
                        #print('Kernel: {}'.format(Array_new[i:l + i, j:l + j, k:l + k]))
                        #print('Qs: {}'.format(Qs[Index]))
                        #print('\n')
                        #print('\n')

                        if(np.array_equal(np.array(Array_new[i:l + i, j:l + j, k:l + k]), np.array(Qs[Index]))):
                            Qs_value[Index] += 1
                            print('Q{}_value: {}'.format(Index, Qs_value[Index]))

                    print(Qs_value)
                    print('\n')

        #           
        List_string = ''

        for i in range(256):
            List_string = List_string + str(Qs_value[i]) + ', '

        print('[{}]'.format(List_string))

        return Qs_value

Array = get_octovoxel_3D(r'Objects\3D\Images backup\Image_random_11_3D.txt')
#Array = get_octovoxel_3D(r'Objects\3D\Images\Image_random_0_3D.txt')

Dataframe = pd.read_csv(r"Objects\3D\Data\Dataframe_test.csv")

# * Return a dataframe with only the data without the labels
X = Dataframe.iloc[:, 1:257].values

# * Return a dataframe with only the labels
Y = Dataframe.iloc[:, -1].values

#X = np.expand_dims(X, axis = 1)
Y = np.expand_dims(Y, axis = 1)

Array = np.expand_dims(Array, axis = 0)

print(X)
print(Y)

print(X.shape)
print(X.shape[1])
print(Y.shape)
print(Array.shape)


# *
Model_RF = RandomForestClassifier(  criterion = 'gini',
                                    n_estimators = 20,
                                    random_state = 2,
                                    n_jobs = 20)

# *
print('\n')
print("Training...")
print('\n')

# *
Model_RF.fit(X, Y)

print('\n')
print("Model trained")
print('\n')

pred_Input = Model_RF.predict(Array)

print('Prediction output')
print(pred_Input)