import tensorflow as tf
from helper import img_loader as imgloader
import numpy as np

data_train_val_dir = 'characters'
data_set = imgloader.get_dataset(data_train_val_dir)

imgtrain_list, lbltrain_list, imgval_list, lblval_list = imgloader.get_image_and_labels(data_set, 0.99, 0.1)

n_classes = 62
onehot = np.zeros((len(imgtrain_list), n_classes))
onehot[np.arange(len(imgtrain_list)), lbltrain_list] = 1
lbltrain_list = onehot

onehot = np.zeros((len(imgval_list), n_classes))
onehot[np.arange(len(imgval_list)), lblval_list] = 1
lblval_list = onehot
