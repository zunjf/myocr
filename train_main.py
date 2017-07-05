import tensorflow as tf
from helper import img_loader as imgloader
from helper import model_ocr as model
import numpy as np
import config.default as config
import random

data_train_val_dir = 'characters' # directory data

data_set = imgloader.get_dataset(data_train_val_dir)
imgtrain_list, lbltrain_list, imgval_list, lblval_list = imgloader.get_image_and_labels(data_set, 0.8, 0.2)

onehot = np.zeros((len(imgtrain_list), config.n_classes))
onehot[np.arange(len(imgtrain_list)), lbltrain_list] = 1
lbltrain_list = onehot

# Randomize training data
combined = list(zip(imgtrain_list, lbltrain_list))
random.shuffle(combined)
imgtrain_list, lbltrain_list = zip(*combined)

# Labeling
onehot = np.zeros((len(imgval_list), config.n_classes))
onehot[np.arange(len(imgval_list)), lblval_list] = 1
lblval_list = onehot

sess = tf.InteractiveSession()
y = tf.nn.softmax(model.logits)
saver = tf.train.Saver()

# Cross Entropy and accuracy
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=model.logits, labels=model.y_)
loss = tf.reduce_mean(cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(model.y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Summary
tf.summary.scalar('cross_entropy', loss)
tf.summary.scalar('accuracy', accuracy)

# Tensorboard
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train_board', sess.graph)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

itr = 0
for i in range(config.epochs):
    steps = int(len(imgtrain_list) // config.batch_size)
    for step in xrange(steps):
        offset = (step * config.batch_size) % (len(imgtrain_list) - config.batch_size)
        batch_paths = imgtrain_list[offset:(offset + config.batch_size)]
        batch_labels = lbltrain_list[offset:(offset + config.batch_size)]

        batch_data = imgloader.read_data(batch_paths, config.image_size)
        #print np.array(batch_data).shape

        acc, l, sum_merged = sess.run([accuracy, loss, merged], feed_dict={model.x: batch_data,
                                                       model.y_:batch_labels,
                                                       model.keep_prob:0.5})

        itr +=1
        train_writer.add_summary(sum_merged, itr)
        print("epoch %d: %d/%d - loss: %.3f - acc: %.3f" % (i, step, steps, l, acc))
