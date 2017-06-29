import os
from scipy import misc
import numpy as np
import sys

def get_image_and_labels(dataset, portion_train, portion_val):
    imgtrn = []
    lbltrn = []
    imgval = []
    lblval = []

    for i in range(len(dataset)):
        img_paths_flat = dataset[i].image_paths

        lbls_flat = [i] * len(dataset[i].image_paths)
        imgtrn += img_paths_flat[:int(len(img_paths_flat) * portion_train)]
        lbltrn += lbls_flat[:int(len(img_paths_flat) * portion_train)]
        imgval += img_paths_flat[-int(len(img_paths_flat) * portion_val):]
        lblval += lbls_flat[-int(len(img_paths_flat) * portion_val):]

    return imgtrn, lbltrn, imgval, lblval

def prehiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.substract(x, mean), 1/std_adj)

    return y

def read_data(image_list, image_size):
    image_batch = []
    for path in image_list:
        im = misc.imread(path)
        im = misc.imresize(im, (image_size, image_size))
        image_batch.append(prewhiten(im))

    return image_batch

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', '+str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)

def get_dataset(paths):
    dataset = []

    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)

        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            chardir = os.path.join(path_exp, class_name)
            if os.path.isdir(chardir):
                images = os.listdir(chardir)
                image_paths = [os.path.join(chardir, img) for img in images]
                dataset.append(ImageClass(class_name, image_paths))

    return dataset

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y
