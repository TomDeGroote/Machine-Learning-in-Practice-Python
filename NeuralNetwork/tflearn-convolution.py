import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tflearn.datasets.mnist as mnist

# This file shows an abstraction of the convolution implementation used in tensorflow-cnn.py

# Load the mnist data
X, Y, test_x, test_y = mnist.load_data(one_hot=True)

# Reshaping
X = X.reshape([-1, 28, 28, 1])
test_x = test_x.reshape([-1, 28, 28, 1])

# Building the CNN
# Input layer
convnet = input_data(shape=[None, 28, 28, 1], name='input')

# 2 layers of convolution and pooling
convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)
convnet = conv_2d(convnet, 64, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

# Fully connected layer, with dropout
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

# Output layer
convnet = fully_connected(convnet, 10, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='targets')

# Create the model
model = tflearn.DNN(convnet)

# Train the model
model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}),
          snapshot_step=500, show_metric=True, run_id='mnist')

# Use model to predict
# model.predict(x)

# Save model
# model.save(filename)

# Load model,, you'll still need to set the structure of the model though, since load only restores the weights
# model.load(filename)
