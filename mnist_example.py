import tensorflow as tf
import matplotlib.pyplot as plt

# This dataset is the "Hello World" of Keras
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# matplotlib to plot the 0th index training item
plt.imshow(x_train[0])
plt.show()

# Normalizing data has big impact on results
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Sequential model is feed-forward, default model
model = tf.keras.models.Sequential()
# Flattens the input data to a single array
model.add(tf.keras.layers.Flatten())
# Densely connected, meaning every input is connected to each node?
# 128 = how many neurons in this layer?
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# Output layer for classifications = no. of classifications
# For a probability distribution, use an activation func like softmax
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Train the model
# Optimizer = Updates model based on output of loss
# Loss = degree of error, NN works by minimizing loss
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 1 epoch is a full pass through the training data
model.fit(x_train, y_train, epochs=3)

# Evaluate the model against test data
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)
