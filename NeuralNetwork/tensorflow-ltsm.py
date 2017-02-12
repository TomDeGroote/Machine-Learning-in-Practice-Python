import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell

# This file uses a LSTM to predict the number of a given image. It does this by taking the pixels of an image
# in as sequential input as chunks.

# Our data set
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Some basic variables
hm_epochs = 3
n_classes = 10
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128

# Defining some placeholders in our graph
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def rnn_model(data):
    # Initialise the weights and biases of the rnn layer randomly
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Fit the data to the TensorFlow requirements
    data = tf.transpose(data, [1, 0, 2])
    data = tf.reshape(data, [-1, chunk_size])
    data = tf.split(0, n_chunks, data)

    # Create the LSTM cell
    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)
    # Create the rnn
    outputs, states = rnn.rnn(lstm_cell, data, dtype=tf.float32)

    # Calculate the output
    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    # Produces the initial predictions
    prediction = rnn_model(x)
    # Measure how wrong our predictions are
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    # We use AdamOptimizer as our cost optimisation function
    optimizer = tf.train.AdamOptimizer().minimize(cost)

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
                epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))

                # Run the optimiser and cost function on our data
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            # Print our progress
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss', epoch_loss)

        # Count the number of correct predictions
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        # Check our prediction accuracy on the test data set
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)),
                                          y: mnist.test.labels}))

# Run the training method
train_neural_network(x)

