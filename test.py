import tensorflow as tf
from helper import img_loader as imgloader
from helper import model_ocr as model
import numpy as np
import config.default as config
import os
import sys
import text_segmentation as segment
import shutil

def ocr_testing(model_name):

    # Image segmentation
    shutil.rmtree('./segment/')
    os.makedirs('./segment/')
    segment.horizontal("./ktp_ori.png",0)

    data_train_val_dir = 'segment'
    data_set = imgloader.get_test_dataset(data_train_val_dir)

    y = tf.nn.softmax(model.logits)

    # Cross Entropy and accuracy
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model.logits, labels=model.y_)
    loss = tf.reduce_mean(cross_entropy)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(model.y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    prediction = tf.argmax(y,1)
    label = tf.argmax(model.y_, 1)
    label_map = imgloader.get_classes("characters")

    #test_data = imgloader.read_data(imgval_list, config.image_size)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        if os.path.isfile('model/' + model_name + ".meta"):
            saver = tf.train.import_meta_graph('model/' + model_name + ".meta")
            saver.restore(sess, tf.train.latest_checkpoint('model/'))

            # Get images for each row on the KTP image and predict it
            for i in range(len(data_set)):
                imgval_list = data_set[i].image_paths
                test_data = imgloader.read_data(imgval_list, config.image_size)
                predictions = sess.run(prediction, feed_dict={model.x:test_data})
                for i in range(len(predictions)):
                    print(label_map[predictions[i]], end = '')
                print('')

if __name__ == "__main__":
    ocr_testing(sys.argv[1])
