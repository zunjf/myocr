import tensorflow as tf
from helper import img_loader as imgloader
from helper import model_ocr as model
import numpy as np
import config.default as config
import random

data_train_val_dir = 'characters' # directory data

data_set = imgloader.get_dataset(data_train_val_dir)
_, _, imgval_list, lblval_list = imgloader.get_image_and_labels(data_set, 0.8, 0.2)

# Labeling
onehot = np.zeros((len(imgval_list), config.n_classes))
onehot[np.arange(len(imgval_list)), lblval_list] = 1
lblval_list = onehot

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
saver = tf.train.import_meta_graph('model/ocr_model_new.meta')
saver.restore(sess, tf.train.latest_checkpoint('model/'))
print("Model Restored.")

y = tf.nn.softmax(model.logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model.logits, labels=model.y_)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(model.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

test_data = imgloader.read_data(imgval_list[0:100], config.image_size)
print("test accuracy %g"%accuracy.eval(feed_dict={model.x: test_data, model.y_: lblval_list[0:100], model.keep_prob: 1.0}))
