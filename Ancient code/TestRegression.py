import tensorflow as tf
import pandas as pd
import numpy as np
from Article_Euler_Number_Info_3D_General import *

matplotlib.use("TkAgg")

from Article_Euler_Number_Class_EulerImageGenerator import DataEuler
from Article_Euler_Number_2D_And_3D_ML import Utilities
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML2D
from Article_Euler_Number_2D_And_3D_ML import EulerNumberML3D

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

        l = 2

        # * Saving the function into a varible array
        Array_new = read_image_with_metadata_3D(Object);

        # * Extract the truth table
        Qs = table_binary_multi_256(256);

        # * Create a empty numpy array
        Qs_value = np.zeros((256), dtype = 'int');

        # * From each combination of the truth table, subtract the number of times each octovovel combination is presented.
        for i in range(Array_new.shape[0] - 1):
            for j in range(Array_new.shape[1] - 1):
                for k in range(Array_new.shape[2] - 1):

                    for Index in range(len(Qs)):
                        
                        # * Compare arrays with different dimensions.
                        if(np.array_equal(np.array(Array_new[i:l + i, j:l + j, k:l + k]), np.array(Qs[Index]))):
                            Qs_value[Index] += 1;
                            print('Q{}_value: {}'.format(Index, Qs_value[Index]));

                    # * print the difference between arrays
                    print(Qs_value)
                    print('\n')
        
        List_string = ''

        for i in range(256):

            List_string = List_string + str(Qs_value[i]) + ', ';
            if(i == 255):
                List_string = List_string + str(Qs_value[i]) + ', ';

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

x = Array
y = np.arange(len(Array))

print(X.shape[1])
print(y)
"""
#plt.hist(Array, len(Array))
plt.bar(y, x, align = 'center')
plt.xlabel('Octovoxeles')
plt.ylabel('Cantidad')
plt.ylim(0, 300)
plt.show()
plt.close()

Array = np.expand_dims(Array, axis = 0)


print(X)
print(Y)

print(X.shape)
print(X.shape[1])
print(Y.shape)
print(Array.shape)


# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape = X.shape[1],))
model.add(tf.keras.layers.Dense(units = 1200, activation = 'relu', kernel_initializer = 'normal'))
model.add(tf.keras.layers.Dense(units = 1))

adamOpti = RMSprop(lr = 0.000001)
rmsprop = 'rmsprop'
# Compile the model
model.compile(optimizer = adamOpti, loss = 'mean_squared_error')

# Provide the data

#print(tf.expand_dims(X, axis = -1))
#print(tf.expand_dims(X, axis = 0))

# Train the model
model.fit(X, Y, batch_size = 8, epochs = 6000)

print('\n')
print("Model trained")
print('\n')

# * Saving model using .h5
Model_name_save = 'Test_euler_rms.h5'.format()
Model_folder_save = os.path.join(r'Objects\3D\Data', Model_name_save)

model.save(Model_folder_save)

Model_prediction = load_model(r'Objects\3D\Data\Test_euler.h5')
#Result = np.argmax(Model_prediction.predict(Array), axis = 1)
#print(model.predict(Array))
print(Model_prediction.predict(Array))
"""
#print(len(Array))

#Result = np.argmax(model.predict(Array), axis = 1)

#result = model.evaluate(Array)

# Make predictions
