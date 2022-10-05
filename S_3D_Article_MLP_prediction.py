import joblib
import numpy as np

from tensorflow.keras.models import load_model
from MLP_2D_3D import Predictions_3D

def prediction_MLP_3D():

    Array = np.loadtxt(r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_3D_1.txt", delimiter = ',')

    print(Array.shape[0])
    print(Array.shape[1])

    Height = Array.shape[0]/Array.shape[1]

    Array_new = Array.reshape(int(Height), int(Array.shape[1]), int(Array.shape[1]))

    print(Array_new)

    print(Array_new.shape[0])
    print(Array_new.shape[1])
    print(Array_new.shape[2])

    Array_quad = np.zeros((2, 2, 2))
    Array_prediction = np.zeros((8))

    #print(Array_prediction)

    Arrays = []

    MLP_result_3D = 0

    Asterisks = 30
    #Model = load_model('Model_3D.h5')
    with open("forest.joblib", 'rb') as f:
        Model_rf = joblib.load(f)

    for i in range(Array_new.shape[0] - 1):
        for j in range(Array_new.shape[1] - 1):
            for k in range(Array_new.shape[2] - 1):

                Array_quad[0][0][0] = Array_new[i][j][k]
                Array_quad[0][0][1] = Array_new[i][j][k + 1]

                Array_quad[0][1][0] = Array_new[i][j + 1][k]
                Array_quad[0][1][1] = Array_new[i][j + 1][k + 1]

                Array_quad[1][0][0] = Array_new[i + 1][j][k]
                Array_quad[1][0][1] = Array_new[i + 1][j][k + 1]

                Array_quad[1][1][0] = Array_new[i + 1][j + 1][k]
                Array_quad[1][1][1] = Array_new[i + 1][j + 1][k + 1]

                Array_prediction[0] = Array_new[i + 1][j][k]
                Array_prediction[1] = Array_new[i + 1][j][k + 1]

                Array_prediction[2] = Array_new[i][j][k]
                Array_prediction[3] = Array_new[i][j][k + 1]

                Array_prediction[4] = Array_new[i + 1][j + 1][k]
                Array_prediction[5] = Array_new[i + 1][j + 1][k + 1]

                Array_prediction[6] = Array_new[i][j + 1][k]
                Array_prediction[7] = Array_new[i][j + 1][k + 1]

                #print(Array_quad)
                print('\n')
                print("*" * Asterisks)

                Array_prediction_list = Array_prediction.tolist()
                Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                True_result_3D = Predictions_3D(Model_rf, Array_prediction) ####
                
                MLP_result_3D += True_result_3D

                print("Kernel array")
                print(Array_quad)
                print('\n')
                print("Prediction array")
                print(Array_prediction)
                Arrays.append('{} -------------- {}'.format(Array_prediction_list_int, True_result_3D))
                print('\n')

                print("*" * Asterisks)

    Array_quad[0][0][0] = 1 #'a'
    Array_quad[0][0][1] = 2 #'c'
    Array_quad[0][1][0] = 3 #'b'
    Array_quad[0][1][1] = 4 #'d'
    Array_quad[1][0][0] = 5 #'e'
    Array_quad[1][0][1] = 6 #'h'
    Array_quad[1][1][0] = 7 #'f'
    Array_quad[1][1][1] = 8 #'g'
    
    print('\n')

    print(Array_quad)

    print('\n')
    
    for i in range(len(Arrays)):
        print('{} ---- {}'.format(i, Arrays[i]))

    print('Euler: {}'.format(MLP_result_3D))