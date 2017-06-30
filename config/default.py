import tensorflow as tf

# Configuration
n_classes = 62
epochs = 100
batch_size = 16
image_size = 24
dropout = 0.75

# Store layers weight & bias
weights = {
           # 5x5, 1 input (channel), 32 outputs
           'wc1' : tf.Variable(tf.random_normal([5,5,3,32])),

           # 5x5 32 input (channel), 64 outputs
           'wc2' : tf.Variable(tf.random_normal([5,5,32,64])),

           # fully connected
           'wd1' : tf.Variable(tf.random_normal([6*6*64, 1024])),

           # 1024 inputs, 10 outpus (class prediction)
           'out' : tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
          'bc1' : tf.Variable(tf.random_normal([32])),
          'bc2' : tf.Variable(tf.random_normal([64])),
          'bd1' : tf.Variable(tf.random_normal([1024])),
          'out' : tf.Variable(tf.random_normal([n_classes]))
}
