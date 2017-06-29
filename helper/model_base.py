import tensorflow as tf

def weight_variable(shape, varname):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W_"+varname)

def bias_variable(shape, varname):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="b_"+varname)

# def conv2d(x, W, stride, padd):
#     return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padd)

def maxpool(h, kernel_sz, stride, padd):
    return tf.nn.max_pool(h, ksize=[1, kernel_sz, kernel_sz, 1], strides=[1, stride, stride, 1], padding=padd)

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)

        return f1 * x + f2 * abs(x)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')
