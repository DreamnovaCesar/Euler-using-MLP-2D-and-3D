
# ?
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ?
from tensorflow.keras.models import load_model

# ?
from keras.models import Sequential
from keras.layers import Dense

# ?
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adamax

# ?
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications import MobileNetV3Large



def model_euler_3D(Euler_number:np.ndarray, Result:np.ndarray, Epochs: int, Model_name:str):

    Model = Sequential()
    Model.add(Dense(units = 1, input_shape = [8]))

    Model.add(Dense(64, activation = "sigmoid"))
    Model.add(Dense(64, activation = "sigmoid"))

    Model.add(Dense(4, activation = 'softmax'))

    #SGD
    Opt = Adam(learning_rate = 0.001)

    Model.compile(
        optimizer = Opt, 
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
        #sparse_categorical_crossentropy
    )   

    print('\n')
    print("Comenzando entrenamiento...")
    print('\n')

    Hist_data = Model.fit(Euler_number, Result, epochs = Epochs, verbose = False)

    print('\n')
    print("Modelo entrenado")
    print('\n')

    Model_name_save = '{}.h5'.format(Model_name)
    Model.save(Model_name_save)

    print("Saved model to disk")
    print('\n')

    Loss = Hist_data.history['loss']
    Accuracy = Hist_data.history['accuracy']

    for i in range(len(Loss)):

        print('{} ------- {}'.format(Loss[i], Accuracy[i]))
        print('\n')

    #print(Loss)
    print('\n')

    return Hist_data

def plot_data(Hist_data):

    #plt.figure(figsize = (20, 20))
    plt.xlabel ("# Epoch")
    plt.ylabel ("# Loss")
    plt.plot(Hist_data.history["loss"])
    plt.show()

"""
def true_data(Result):

    if Result > 0.5:
        New_Result = 1
    elif Result < 0.5 and Result > -0.5:
        New_Result = 0
    elif Result < -0.5:
        New_Result = -1

    return New_Result
"""

def Predictions(Model, Prediction_value):

    #Asterisks = 30

    print("Prediction!")
    #Do not use Model.predict, use model instead
    Result = Model([Prediction_value])

    True_result = true_data(Result)

    #print("*" * Asterisks)
    print('The result is: {}'.format(Result))
    print('The true value is: {}'.format(True_result))
    #print("*" * Asterisks)
    print('\n')

    return True_result

def true_data_3D(Result):

    if Result == 0:
        New_Result = 0
    elif Result == 1:
        New_Result = 1
    elif Result == 2:
        New_Result = -1
    elif Result == 3:
        New_Result = -2

    return New_Result

def Predictions_3D(Model, Prediction_value):

    #Asterisks = 30

    print("Prediction!")
    #Do not use Model.predict, use model instead
    #Result = Model([Prediction_value])

    Result = Model.predict([Prediction_value])

    print(Result)

    #Result = np.argmax(Model.predict([Prediction_value]), axis = 0)

    #print(Result)

    True_result = true_data_3D(Result)

    #print("*" * Asterisks)
    print('The result is: {}'.format(Result))
    print('The true value is: {}'.format(True_result))
    #print("*" * Asterisks)
    print('\n')

    return True_result