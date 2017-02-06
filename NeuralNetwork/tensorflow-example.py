# TensorFlow is a package that allows us to work with tensors very efficiently
import tensorflow as tf

# Creates nodes in a graph = the construction phase
# Creates a constant value node equal to 5
x1 = tf.constant(5)
# Creates a constant value node equal to 6
x2 = tf.constant(6)

# Create a multiplication operation node in our graph
result = tf.mul(x1, x2)
# As shown it is still just an abstract graph, the multiplication has not been executed yet
print("Our abstract graph", result)

# First we build the graph, next we launch it
sess = tf.Session()
# Run the result operation from above
print("The result", sess.run(result))

# Close the session when finished
sess.close()
