
# ?
import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# ?
from typing import Any

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
from sklearn.ensemble import RandomForestClassifier

from S_3D_Article_NU import Array_new

# ?
class EulerNumberML:

    def __init__(self, **kwargs) -> None:
        """
        _summary_

        _extended_summary_
        """
        # * General parameters
        self.Input = kwargs.get('input', None)
        self.Output = kwargs.get('output', None)
        self.Object = kwargs.get('object', None)

        # *
        self.Folder = kwargs.get('folder', None)
        self.Model_name = kwargs.get('modelname', None)
        self.Epochs = kwargs.get('epochs', None)
        self.Columns = ["Loss", "Accuracy"]

    # ?
    @staticmethod
    def print_octovoxel_order() -> None:
        
        Letters = ('a', 'c', 'b', 'd', 'e', 'h', 'f', 'g')
        # *
        Array_prediction_octov = np.zeros((2, 2, 2))

        # *
        Array_prediction_octov[0][0][0] = 1 #'a'
        Array_prediction_octov[0][0][1] = 2 #'c'
        Array_prediction_octov[0][1][0] = 3 #'b'
        Array_prediction_octov[0][1][1] = 4 #'d'
        Array_prediction_octov[1][0][0] = 5 #'e'
        Array_prediction_octov[1][0][1] = 6 #'h'
        Array_prediction_octov[1][1][0] = 7 #'f'
        Array_prediction_octov[1][1][1] = 8 #'g'

        print('\n')
        print(Array_prediction_octov)
        print('\n')

        for i, letter in enumerate(Letters):
            print('{} ------> {}'.format(i + 1, letter))
        print('\n')

    # ?
    def plot_data_loss(self, Hist_data: Any) -> None:
        """
        _summary_

        _extended_summary_

        Args:
            Hist_data (Any): _description_
        """
        #plt.figure(figsize = (20, 20))
        plt.title('Training loss')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Loss")
        plt.plot(Hist_data.history["loss"])
        #plt.show()

        Figure_name = "Figure_Loss_{}.png".format(self.Model_name)
        Figure_name_folder = os.path.join(self.Folder, Figure_name)

        plt.savefig(Figure_name_folder)

    # ?
    def plot_data_accuracy(self, Hist_data: Any) -> None:
        """
        _summary_

        _extended_summary_

        Args:
            Hist_data (Any): _description_
        """
        #plt.figure(figsize = (20, 20))
        plt.title('Training accuracy')
        plt.xlabel ("# Epoch")
        plt.ylabel ("# Acuracy")
        plt.plot(Hist_data.history["accuracy"])
        #plt.show()

        Figure_name = "Figure_Accuracy_{}.png".format(self.Model_name)
        Figure_name_folder = os.path.join(self.Folder, Figure_name)

        plt.savefig(Figure_name_folder)

    # ? Create dataframes
    def create_dataframe_history(self, Column_names: Any, Folder_save: str, CSV_name: str, Hist_data: Any) -> None: 

        # * Lists
        #Column_names = ['Folder', 'New Folder', 'Animal', 'Label']
        
        # *
        Dataframe_created = pd.DataFrame(columns = Column_names)

        # *
        Accuracy = Hist_data.history["accuracy"]
        Loss = Hist_data.history["loss"]
        
        History_data = zip(Loss, Accuracy)

        # *
        for i, (l, a) in enumerate(History_data):
            Dataframe_created.loc[len(Dataframe_created.index)] = [l, a]

        # *
        Dataframe_name = "Dataframe_{}_Loss_And_Accuracy.csv".format(CSV_name)
        Dataframe_folder = os.path.join(Folder_save, Dataframe_name)

        # *
        Dataframe_created.to_csv(Dataframe_folder)

    # ?
    def read_image_with_metadata(self, Array_file) -> Any:
        """
        _summary_

        _extended_summary_

        Args:
            Array_file (_type_): _description_
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

    # ?
    def true_data(self, Result: int):
        """
        _summary_

        _extended_summary_

        Args:
            Result (int): _description_

        Returns:
            _type_: _description_
        """
        if Result > 0.5:
            New_Result = 1
        elif Result < 0.5 and Result > -0.5:
            New_Result = 0
        elif Result < -0.5:
            New_Result = -1

        return New_Result

    # ?
    def true_data_3D(self, Result: int):
        """
        _summary_

        _extended_summary_

        Args:
            Result (int): _description_

        Returns:
            _type_: _description_
        """
        if Result == 0:
            New_Result = 0
        elif Result == 1:
            New_Result = 1
        elif Result == 2:
            New_Result = -1
        elif Result == 3:
            New_Result = -2

        return New_Result

    # ?
    def Predictions(self, Model: Any, Prediction_value: Any) -> int:
        """
        _summary_

        _extended_summary_

        Args:
            Model (Any): _description_
            Prediction_value (Any): _description_

        Returns:
            int: _description_
        """
        #Asterisks = 30

        print("Prediction!")
        #Do not use Model.predict, use model instead
        Result = Model.predict([Prediction_value])

        True_result = self.true_data(Result)

        #print("*" * Asterisks)
        print('The result is: {}'.format(Result))
        print('The true value is: {}'.format(True_result))
        #print("*" * Asterisks)
        print('\n')

        return True_result

    # ? #Model_prediction, Array
    def Predictions_3D(self, Model_name: str, Model_prediction: Any, Prediction_value: Any) -> int:
        """
        _summary_

        _extended_summary_

        Args:
            Model (Any): _description_
            Prediction_value (Any): _description_

        Returns:
            int: _description_
        """
        #Asterisks = 30

        print("Prediction!")
        #Do not use Model.predict, use model instead
        #Result = Model([Prediction_value])

        if Model_name.endswith('.h5'):
            Result = np.argmax(Model_prediction.predict([Prediction_value]), axis = 1)
            print(Result)

        elif Model_name.endswith('.joblib'):
            Result = Model_prediction.predict([Prediction_value])
            print(Result)

        #Result = np.argmax(Model.predict([Prediction_value]), axis = 0)

        #print(Result)

        True_result = self.true_data_3D(Result)

        #print("*" * Asterisks)
        print('The result is: {}'.format(Result))
        print('The true value is: {}'.format(True_result))
        #print("*" * Asterisks)
        print('\n')

        return True_result

    # ?
    def model_euler_2D_MLP(self) -> Any:
        """
        _summary_

        _extended_summary_

        Returns:
            Any: _description_
        """
        print(self.Input.shape)
        print(self.Output.shape)

        Model = Sequential()
        Model.add(Dense(units = 1, input_shape = [4]))
        Model.add(Dense(9, activation = "sigmoid"))
        Model.add(Dense(1, activation = 'tanh'))

        Opt = Adam(learning_rate = 0.1)

        Model.compile(
            optimizer = Opt, 
            loss = 'mean_squared_error'
        )

        print('\n')
        print("Model trained")
        print('\n')
        
        # *
        Hist_data = Model.fit(self.Input, self.Output, epochs = self.Epochs, verbose = False)

        print('\n')
        print("Modelo entrenado")
        print('\n')

        # *
        Model_name_save = '{}.h5'.format(self.Model_name)
        Model.save(Model_name_save)

        print("Saved model to disk")
        print('\n')

        # *
        self.plot_data_loss(Hist_data)
        
        # *
        self.plot_data_accuracy(Hist_data)
        print('\n')

        return Hist_data

    # ?
    def model_euler_3D_MLP(self) -> Any:
        """
        _summary_

        _extended_summary_

        Returns:
            Any: _description_
        """
        # *
        print(self.Input.shape)
        print(self.Output.shape)

        # *
        Model = Sequential()
        Model.add(Dense(units = 1, input_shape = [8]))
        Model.add(Dense(64, activation = "sigmoid"))
        Model.add(Dense(4, activation = 'softmax'))

        # *
        Opt = Adam(learning_rate = 0.001)

        # *
        Model.compile(
            optimizer = Opt, 
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy']
            #sparse_categorical_crossentropy
        )   

        # *
        print('\n')
        print("Training...")
        print('\n')

        # *
        Hist_data = Model.fit(self.Input, self.Output, epochs = self.Epochs, verbose = False)

        print('\n')
        print("Model trained")
        print('\n')

        # * Saving model using .h5
        Model_name_save = '{}.h5'.format(self.Model_name)
        Model.save(Model_name_save)

        # *
        print("Saving model...")
        print('\n')

        # *
        #Loss = Hist_data.history['loss']
        #Accuracy = Hist_data.history['accuracy']

        # *
        self.create_dataframe_history(self.Columns, self.Folder, self.Model_name, Hist_data)

        # *
        self.plot_data_loss(Hist_data)

        # *
        self.plot_data_accuracy(Hist_data)
        print('\n')

        return Hist_data

    # ?
    def model_euler_3D_RF(self) -> None:
        """
        _summary_

        _extended_summary_

        Returns:
            Any: _description_
        """
        # *
        print(self.Input.shape)
        print(self.Output.shape)

        # *
        Model_RF = RandomForestClassifier(  criterion = 'gini',
                                            n_estimators = 10,
                                            random_state = 2,
                                            n_jobs = 10)

        # *
        print('\n')
        print("Training...")
        print('\n')

        # *
        Model_RF.fit(self.Input, self.Output)

        print('\n')
        print("Model trained")
        print('\n')

        # * Saving model using .h5
        Model_name_save = '{}.joblib'.format(self.Model_name)
        joblib.dump(Model_RF, Model_name_save)

        # *
        print("Saving model...")
        print('\n')

        # *
        #Loss = Hist_data.history['loss']
        #Accuracy = Hist_data.history['accuracy']

        # *
        #self.create_dataframe_history(self.Columns, self.Folder, self.Model_name, Hist_data)

        # *
        #self.plot_data_loss(Hist_data)

        # *
        #self.plot_data_accuracy(Hist_data)
        print('\n')

    # ?
    def obtain_arrays_from_object(self):

        #Array = np.loadtxt(self.Object, delimiter = ',')

        # *
        Arrays = []
        Asterisks = 30

        # *
        Array_new = self.read_image_with_metadata(self.Object)

        # *
        Array_prediction_octov = np.zeros((2, 2, 2))
        Array_prediction = np.zeros((8))

        # *
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    # *
                    Array_prediction_octov[0][0][0] = Array_new[i][j][k]
                    Array_prediction_octov[0][0][1] = Array_new[i][j][k + 1]

                    Array_prediction_octov[0][1][0] = Array_new[i][j + 1][k]
                    Array_prediction_octov[0][1][1] = Array_new[i][j + 1][k + 1]

                    Array_prediction_octov[1][0][0] = Array_new[i + 1][j][k]
                    Array_prediction_octov[1][0][1] = Array_new[i + 1][j][k + 1]

                    Array_prediction_octov[1][1][0] = Array_new[i + 1][j + 1][k]
                    Array_prediction_octov[1][1][1] = Array_new[i + 1][j + 1][k + 1]

                    # *
                    Array_prediction[0] = Array_new[i + 1][j][k]
                    Array_prediction[1] = Array_new[i + 1][j][k + 1]

                    Array_prediction[2] = Array_new[i][j][k]
                    Array_prediction[3] = Array_new[i][j][k + 1]

                    Array_prediction[4] = Array_new[i + 1][j + 1][k]
                    Array_prediction[5] = Array_new[i + 1][j + 1][k + 1]

                    Array_prediction[6] = Array_new[i][j + 1][k]
                    Array_prediction[7] = Array_new[i][j + 1][k + 1]
                    print('\n')

                    # *
                    print("*" * Asterisks)
                    Array_prediction_list = Array_prediction.tolist()
                    Array_prediction_list_int = [int(i) for i in Array_prediction_list]

                    # *
                    print("Kernel array")
                    print(Array_prediction_octov)
                    print('\n')
                    print("Prediction array")
                    print(Array_prediction)
                    print('\n')
                    Arrays.append(Array_prediction_list_int)
                    print("*" * Asterisks)
                    print('\n')

        # *
        for i in range(len(Arrays)):
            print('{} ---- {}'.format(i, Arrays[i]))
        print('\n')
        
        return Arrays

    # ?
    def model_prediction_3D(self, Model, Arrays):

        #Array = np.loadtxt(r"C:\Users\Cesar\Dropbox\PC\Desktop\MLP_article_2D\Example_3D_1.txt", delimiter = ',')

        # *
        Prediction_result_3D = 0
        Asterisks = 30

        # * Read multilayer perceptron model
        if Model.endswith('.h5'):
            Model_prediction = load_model(Model)

        # * Read machine learning model
        elif Model.endswith('.joblib'):
            Model_prediction = joblib.load(Model)

        # *
        for i, Array in enumerate(Arrays):

            True_result_3D = self.Predictions_3D(Model, Model_prediction, Array) ####
            Prediction_result_3D += True_result_3D

            print('{} -------------- {}'.format(Array, True_result_3D))
        print('\n')

        print('Euler: {}'.format(Prediction_result_3D))
        print('\n')