import numpy as np
from MLP_2D_3D import Predictions_3D
from tensorflow.keras.models import load_model

Array_prediction = np.zeros((1, 8))

Array_prediction[0][0] = 0
Array_prediction[0][1] = 0
Array_prediction[0][2] = 0
Array_prediction[0][3] = 1
Array_prediction[0][4] = 1
Array_prediction[0][5] = 0
Array_prediction[0][6] = 0
Array_prediction[0][7] = 1

print(Array_prediction.shape)

print(Array_prediction)

Model = load_model('Model_3D.h5')

Result = Model([Array_prediction])

Result = np.argmax(Model([Array_prediction]), axis = 1)

print(Result)
print(Result.dtype)

if Result == 2:
    Result = -1

print(Result)
#True_result_3D = Predictions_3D(Model, Array_prediction) ####