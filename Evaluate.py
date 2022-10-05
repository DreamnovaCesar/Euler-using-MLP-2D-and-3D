import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


results = model.evaluate(x_test, y_test, batch_size = 128)
print("test loss, test acc:", results)