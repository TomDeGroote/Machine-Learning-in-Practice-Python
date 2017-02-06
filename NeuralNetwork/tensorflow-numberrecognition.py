import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# In this file we use the MNIST data set and TensorFlow to predict the number of a given
# written number from 0 to 9.

# Our data set
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Start building our model by defining the number of nodes in every hidden layer
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
# The number of classes, 0 to 9 so 10 classes
n_classes = 10
# Batch size to train the network
batch_size = 100

# Defining some placeholders in our graph, 784 because we are working on 28 by 28 pixel images (it defines
# the shape of the variable for TensorFlow)
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')


def neural_network_model(data):
    # Defines the weights and biases for the network randomly
    # biases here are a value that is added to our sums (so that our network still works if all nodes fire
    # a 0), these biases will need to be op
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}

    # Here we start the feed forward flow
    # layer 1 = input data * weights + biases
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    # layer 2 = layer 1 * weights + biases
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)
    # layer 3 = layer 2 * weights + biases
    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)
    # output = layer 3 * weights + biases
    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    # Produces the initial predictions
    prediction = neural_network_model(x)
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

