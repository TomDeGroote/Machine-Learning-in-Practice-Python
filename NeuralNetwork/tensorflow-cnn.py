import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In this file we use the MNIST data set and TensorFlow to predict the number of a given
# written number from 0 to 9.

# Our data set
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# The number of classes, 0 to 9 so 10 classes
n_classes = 10
# Batch size to train the network
batch_size = 128

# The rate at which we drop neurons
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

# Defining some placeholders in our graph, 784 because we are working on 28 by 28 pixel images (it defines
# the shape of the variable for TensorFlow)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, n_classes])


# Convolution
def conv2d(x, W):
    # The stride parameter dictates the movement of the window, in this case one pixel at a time
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# Pooling
def maxpool2d(x):
    # The stride parameter dictates the movement of the window, in this case two pixels at a time
    # ksize is the size of the pooling window, in this case a 2x2 window for pooling
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def cnn_model(data):
    # Define the weights
    weights = {
        # 5x5 convolution, 1 input image, 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        # 5x5 convolution, 32 input images, 64 outputs
        'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        # Fully connected 7*7*64 inputs, 1024 outputs
        'W_fc': tf.Variable(tf.random_normal([7*7*64, 1024])),
        # Output layer 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    # Define biases
    biases = {
        'b_conv1': tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Reshape input to 4D tensor
    data = tf.reshape(data, shape=[-1, 28, 28, 1])
    # Convolution Layer, using our function conv2d
    conv1 = tf.nn.relu(conv2d(data, weights['W_conv1']) + biases['b_conv1'])
    # Max pooling
    conv1 = maxpool2d(conv1)
    # Convolution layer
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    # Fully connected layer
    fc = tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    # Dropping some neurons
    fc = tf.nn.dropout(fc, keep_rate)
    # Output layer
    output = tf.matmul(fc, weights['out']) + biases['out']

    return output


def train_neural_network(x):
    # Produces the initial predictions
    prediction = cnn_model(x)
    # Measure how wrong our predictions are
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    # We use AdamOptimizer as our cost optimisation function
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Define the number of epochs
    hm_epochs = 10
    # Start a session
    with tf.Session() as sess:
        # Initialise all variables
        sess.run(tf.global_variables_initializer())

        # Train for the number of epochs
        for epoch in range(hm_epochs):
            # We will keep track of our error
            epoch_loss = 0
            # Train for every batch
            for _ in range(int(mnist.train.num_examples/batch_size)):
                # get the x and y data for this epoch (from the MNIST data set)
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                # Run the optimiser and cost function on our data
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # Print our progress
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

        # Count the number of correct predictions
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # Check our prediction accuracy on the test data set
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

# Run the training method
train_neural_network(x)

