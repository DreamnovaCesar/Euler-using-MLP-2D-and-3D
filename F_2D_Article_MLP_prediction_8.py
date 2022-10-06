import numpy as np
from PIL import Image

#import tensorflow as tf
#tf.compat.v1.disable_eager_execution()

from tensorflow.keras.models import load_model

from Article_Euler_Number_2D_General import read_image_with_metadata

from MLP_2D_3D import Predictions

Asterisks = 30

MLP_result_connected_8 = 0

#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_1.txt"
#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_2.txt"
Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_3.txt"
#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_4.txt"
#Array_MLP = r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_2D_5.txt"

Array = read_image_with_metadata(Array_MLP)

Array_comparison = np.zeros((2, 2), dtype = 'int')
Array_prediction = np.zeros((1, 4), dtype = 'int')

Model = load_model('Model_8_connection.h5')

for i in range(Array.shape[0] - 1):
    for j in range(Array.shape[1] - 1):

        #Array_comparison[0][0] = Array[i][j]
        #Array_comparison[1][0] = Array[i + 1][j]
        #Array_comparison[0][1] = Array[i][j + 1]
        #Array_comparison[1][1] = Array[i + 1][j + 1]

        Array_prediction[0][0] = Array[i][j]
        Array_prediction[0][2] = Array[i + 1][j]
        Array_prediction[0][1] = Array[i][j + 1]
        Array_prediction[0][3] = Array[i + 1][j + 1]

        print('\n')
        #print("*" * Asterisks)

        True_result_connected_8 = Predictions(Model, Array_prediction) ####
        
        MLP_result_connected_8 += True_result_connected_8

        #print("Kernel array")
        #print(Array_comparison)
        #print('\n')
        print("Prediction array")
        print(Array_prediction)
        print('\n')

        #print("*" * Asterisks)

print('Connectivity 8: {}'.format(MLP_result_connected_8))
