import tensorflow as tf

# Configuration
n_classes = 62
epochs = 100
batch_size = 16
image_size = 24
dropout = 0.75

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Store layers weight & bias
weights = {
           # 5x5, 3 input (channel), 32 outputs
           'wc1' : weight_variable([5,5,3,32]),#tf.Variable(tf.random_normal([5,5,3,32])),

           # 5x5 32 input (channel), 64 outputs
           'wc2' : weight_variable([5,5,32,64]),#tf.Variable(tf.random_normal([5,5,32,64])),

           # fully connected
           'wd1' : weight_variable([6*6*64, 1024]),#tf.Variable(tf.random_normal([6*6*64, 1024])),

           # 1024 inputs, 10 outpus (class prediction)
           'out' : weight_variable([1024, n_classes])#tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
          'bc1' : bias_variable([32]),#tf.Variable(tf.random_normal([32])),
          'bc2' : bias_variable([64]),#tf.Variable(tf.random_normal([64])),
          'bd1' : bias_variable([1024]),#tf.Variable(tf.random_normal([1024])),
          'out' : bias_variable([n_classes])#tf.Variable(tf.random_normal([n_classes]))
}
