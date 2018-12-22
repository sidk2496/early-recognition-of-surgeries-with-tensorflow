import tensorflow as tf

def new_weights(shape):
    return tf.get_variable(name='weights', initializer=tf.truncated_normal(shape=shape, stddev=0.05))


def new_biases(length):
    return tf.get_variable(name='biases', initializer=tf.constant(0.05, shape=[length]))


# def conv(input, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME'):
#
#     with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
#
#         num_input_channels = tf.cast(input.get_shape()[3], tf.int32)
#         shape = [filter_height, filter_width, num_input_channels, num_filters]
#         weights = new_weights(shape=shape)
#         biases = new_biases(length=num_filters)
#
#         layer = tf.nn.conv2d(input=input, filter=weights,
#                              strides=[1 , stride_y, stride_x, 1],
#                              padding=padding, name=name)
#         layer+=biases
#         layer = tf.nn.relu(layer)
#
#         return layer, weights


def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name, padding='SAME', groups=1):
    """
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=[1, stride_y, stride_x, 1], padding=padding)

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', initializer=tf.truncated_normal(
                                  shape=[filter_height, filter_width, input_channels // groups, num_filters],
                                  stddev=0.05))
        biases = tf.get_variable('biases', initializer=tf.constant(0.05, shape=[num_filters]))
        if groups == 1:
            conv = convolve(x, weights)


        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)
        return relu, {'weights': weights, 'biases': biases}


def max_pool(input, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(value=input, ksize=[1, filter_height, filter_width, 1],
                              strides=[1, stride_y, stride_x, 1], padding=padding, name=name)


def lrn(input, depth_radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(input=input, depth_radius=depth_radius, alpha=alpha,
                                                  beta=beta, bias=bias, name=name)


def flatten(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def fc(input, num_outputs, name, use_relu=True):
    num_inputs = int(input.get_shape()[1])
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        weights = new_weights(shape=[num_inputs, num_outputs])
        biases = new_biases(length=num_outputs)
        layer = tf.matmul(input, weights) + biases

        if use_relu:
            layer = tf.nn.relu(layer)

        return layer, {'weights': weights, 'biases': biases}


def dropout(input, keep_prob, name):
    return tf.nn.dropout(input, keep_prob=keep_prob, name=name)