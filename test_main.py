import tensorflow as tf
from helper import img_loader as imgloader
from helper import model_ocr as model
import numpy as np
import config.default as config
import os.path
import sys

def ocr_testing(model_name):
    data_train_val_dir = 'characters' # directory data

    data_set = imgloader.get_dataset(data_train_val_dir)
    _, _, imgval_list, lblval_list = imgloader.get_image_and_labels(data_set, 0.8, 0.2)

    # Labeling
    onehot = np.zeros((len(imgval_list), config.n_classes))
    onehot[np.arange(len(imgval_list)), lblval_list] = 1
    lblval_list = onehot

    #sess = tf.InteractiveSession()
    y = tf.nn.softmax(model.logits)

    # Cross Entropy and accuracy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model.logits, labels=model.y_)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(model.y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    test_data = imgloader.read_data(imgval_list, config.image_size)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.isfile('model/' + model_name + ".meta"):
            saver = tf.train.import_meta_graph('model/' + model_name + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint('model/'))
            print('test accuracy %g' % accuracy.eval(feed_dict={model.x: test_data, model.y_: lblval_list, model.keep_prob: 1.0}))

if __name__ == "__main__":
    ocr_testing(sys.argv[1])
