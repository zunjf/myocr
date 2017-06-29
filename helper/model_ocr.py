from model_base import *
import config.default as config

x = tf.placeholder(tf.float32, shape=[None, config.image_size, config.image_size, 1])
y_ = tf.placeholder(tf.float32, shape=[None, config.n_classes])
keep_prob = tf.placeholder(tf.float32)

# W = weight_variable(config.weights['wc1'], "conv1")
# # tf.histogram_summary("W_conv1", W)
# b = bias_variable(config.biases['bc1'], 'conv1')
# conv = conv2d(x, W, 1, 'SAME')
# h = lrelu(tf.nn.bias_add(conv, b))
# net = maxpool(h, 2, 2, 'SAME')
#
# W = weight_variable(config.weights['wc2'], "conv2")
# # tf.histogram_summary("W_conv2", W)
# b = bias_variable(config.biases['bc2'], 'conv2')
# conv = conv2d(net, W, 1, 'SAME')
# h = lrelu(tf.nn.bias_add(conv, b))
# net = maxpool(h, 2, 2, 'SAME')
#
# net = tf.nn.avg_pool(net, ksize=[1, 6, 6, 1], strides=[1, 6, 6, 1], padding='SAME')
# print(net.get_shape())
# fc1 = tf.reshape(net, [-1, config.weights['wd1'].get_shape().as_list()[0]])
# fc1 = tf.add(tf.matmul(fc1, config.weights['wd1']), config.biases['bd1'])
# fc1 = tf.nn.relu(fc1)
# fc1 = tf.nn.dropout(fc1, config.dropout)

x = tf.reshape(x, shape=[-1, config.image_size, config.image_size, 1])

# Convolutional layer
conv1 = conv2d(x, config.weights['wc1'], config.biases['bc1'])
# Max Pooling (down-sampling)
conv1 = maxpool2d(conv1, k=2)

# Convolutional layer
conv2 = conv2d(conv1, config.weights['wc2'], config.biases['bc2'])
# Max Pooling (down-sampling)
conv2 = maxpool2d(conv2, k=2)

# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
fc1 = tf.reshape(conv2, [-1, config.weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, config.weights['wd1']), config.biases['bd1'])
fc1 = tf.nn.relu(fc1)

# Apply Dropout
fc1 = tf.nn.dropout(fc1, config.dropout)

logits = tf.add(tf.matmul(fc1, config.weights['out']), config.biases['out'])
