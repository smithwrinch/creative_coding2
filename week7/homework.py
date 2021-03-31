import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
plt.style.use('ggplot')
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

tf.disable_v2_behavior()


# this function will measure the absolute distance, also known as the l1-norm
def distance(p1, p2):
    return tf.abs(p1 - p2)



def create_toy_data():
    # Let's create some toy data

    # We are going to say that we have seen 1000 values of some underlying representation that we aim to discover
    n_observations = 1000

    # Instead of having an image as our input, we're going to have values from -3 to 3.  This is going to be the input to our network.
    xs = np.linspace(-3, 3, n_observations)

    # From this input, we're going to teach our network to represent a function that looks like a sine wave.  To make it difficult, we are going to create a noisy representation of a sine wave by adding uniform noise.  So our true representation is a sine wave, but we are going to make it difficult by adding some noise to the function, and try to have our algorithm discover the underlying cause of the data, which is the sine wave without any noise.
    ys = np.cos(xs) * np.sin(xs) + np.tanh(xs) + np.random.uniform(-0.5, 0.5, n_observations)
    plt.scatter(xs, ys, alpha=0.15, marker='+')
    # plt.show()
    return xs, ys

def set_up():
    # variables which we need to fill in when we are ready to compute the graph.
    # We'll pass in the values of the x-axis to a placeholder called X.
    X = tf.placeholder(tf.float32, name='X')

    # And we'll also specify what the y values should be using another placeholder, y.
    Y = tf.placeholder(tf.float32, name='Y')
    sess = tf.InteractiveSession()
    n = tf.random_normal([1000], stddev=0.1).eval()
    # plt.hist(n)

    # We're going to multiply our input by 10 values, creating an "inner layer"
    # of n_neurons neurons.
    n_neurons = 10
    W = tf.Variable(tf.random_normal([1, n_neurons]), name='W')

    # and allow for n_neurons additions on each of those neurons
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]), name='b')

    # Instead of just multiplying, we'll put our n_neuron multiplications through a non-linearity, the tanh function.
    h = tf.nn.tanh(tf.matmul(tf.expand_dims(X, 1), W) + b, name='h')

    Y_pred = tf.reduce_sum(h, 1)

    cost = tf.reduce_mean(distance(Y_pred, Y))


    return X, Y, Y_pred, cost


def train(n_iterations=100, batch_size=200, learning_rate=0.02):

    xs, ys = create_toy_data()
    X, Y, Y_pred, cost = set_up()

    cost = tf.reduce_mean(distance(Y_pred, Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(xs, ys, alpha=0.15, marker='+')
    ax.set_xlim([-4, 4])
    ax.set_ylim([-2, 2])

    with tf.Session() as sess:
        # Here we tell tensorflow that we want to initialize all
        # the variables in the graph so we can use them
        # This will set W and b to their initial random normal value.
        sess.run(tf.global_variables_initializer())

        # We now run a loop over epochs
        prev_training_cost = 0.0
        for it_i in range(n_iterations):
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs) // batch_size
            for batch_i in range(n_batches):
                idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})

            if it_i % 10 == 0:
                ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
                ax.plot(xs, ys_pred, 'k', alpha=it_i / n_iterations)
                print(training_cost)
    fig.show()
    plt.draw()
    plt.show()

def simpleY():
    # To create the variables, we'll use tf.Variable, which unlike a placeholder, does not require us to define the value at the start of a run/eval.  It does need an initial value, which we'll give right now using the function tf.random_normal.  We could also pass an initializer, which is simply a function which will call the same function.  We'll see how that works a bit later.  In any case, the random_normal function just says, give me a random value from the "normal" curve.  We pass that value to a tf.Variable which creates a tensor object.
    W = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=0.1), name='weight')

    # For bias variables, we usually start with a constant value of 0.
    B = tf.Variable(tf.constant([0], dtype=tf.float32), name='bias')

    # Now we can scale our input placeholder by W, and add our bias, b.
    Y_pred = X * W + B
    return Y_pred

def simpleY2():
    # with hidden layers


    # We're going to multiply our input by 100 values, creating an "inner layer"
    # of 100 neurons.
    n_neurons = 100
    W = tf.Variable(tf.random_normal([1, n_neurons], stddev=0.1))

    # and allow for n_neurons additions on each of those neurons
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_neurons]))

    # Instead of multiplying directly, we use tf.matmul to perform a
    # matrix multiplication
    h = tf.matmul(tf.expand_dims(X, 1), W) + b

    # Create the operation to add every neuron's output
    Y_pred = tf.reduce_sum(h, 1)

    return Y_pred

def YPol():

    # Instead of a single factor and a bias, we'll create a polynomial function
    # of different degrees.  We will then learn the influence that each
    # degree of the input (X^0, X^1, X^2, ...) has on the final output (Y).
    Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
    for pow_i in range(1, 5):
        W = tf.Variable(
            tf.random_normal([1], stddev=0.1), name='weight_%d' % pow_i)
        Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)


def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer()) # This initializer will create a new random value every time we call sess.run(tf.global_variables_initializer()).
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h


def deep_neural():
    ops.reset_default_graph()

    # let's get the current graph
    g = tf.get_default_graph()

    # See the names of any operations in the graph
    [op.name for op in tf.get_default_graph().get_operations()]

    # let's create a new network
    X = tf.placeholder(tf.float32, name='X')
    h = linear(X, 2, 10)

    h2 = linear(h, 10, 10, scope='layer2')
    # See the names of any operations in the graph
    [op.name for op in tf.get_default_graph().get_operations()]

train()
