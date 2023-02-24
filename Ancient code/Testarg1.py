import tensorflow as tf

# Define the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=2, input_shape=[2]))
model.add(tf.keras.layers.Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Provide the data
x = [[2, 2], [3, 3], [4, 4], [5, 5]]
y = [2, 4, 6, 8]

# Train the model
model.fit(x, y, epochs=500)

# Make predictions
print(model.predict([[6, 6]]))