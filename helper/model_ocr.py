from model_base import *
import config.default as config

x = tf.placeholder(tf.float32, shape=[None, config.image_size, config.image_size, 3])
y_ = tf.placeholder(tf.float32, shape=[None, config.n_classes])
keep_prob = tf.placeholder(tf.float32)

# Convolutional layer
print x.get_shape()
conv1 = conv2d(x, config.weights['wc1'], config.biases['bc1'], 1, 'conv1', True)

# Max Pooling (down-sampling)
print conv1.get_shape()
conv1 = maxpool2d(conv1, k=2)
print conv1.get_shape()

# Convolutional layer
conv2 = conv2d(conv1, config.weights['wc2'], config.biases['bc2'], 1, 'conv2', True)

# Max Pooling (down-sampling)
print conv2.get_shape()
conv2 = maxpool2d(conv2, k=2)
print conv2.get_shape()

# Fully connected layer
# Reshape conv2 output to fit fully connected layer input
print  config.weights['wd1'].get_shape().as_list()[0]
fc1 = tf.reshape(conv2, [-1, config.weights['wd1'].get_shape().as_list()[0]])
fc1 = tf.add(tf.matmul(fc1, config.weights['wd1']), config.biases['bd1'])
fc1 = tf.nn.relu(fc1)

# Apply Dropout
fc1 = tf.nn.dropout(fc1, config.dropout)

logits = tf.add(tf.matmul(fc1, config.weights['out']), config.biases['out'])
