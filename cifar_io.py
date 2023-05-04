import numpy as np
import os

import pickle

img_size = 32 # CIFAR-10 image dimension: 32x32
nImgs_train = 50000
nImgs_test = 10000
nClasses = 10 # 10 different classes in CIFAR-10

path_data = os.path.abspath('data/cifar-10/cifar-10-batches-py')
#path_data = 'data/cifar-10/cifar-10-batches-py'
list_files_train = list([path_data+'/data_batch_1',
                         path_data+'/data_batch_2',
                         path_data+'/data_batch_3',
                         path_data+'/data_batch_4',
                         path_data+'/data_batch_5',])
list_files_test = list([path_data+'/test_batch'])
lsit_files_label_names = list([path_data+'/batches.meta'])

# function taken from: https://www.cs.toronto.edu/~kriz/cifar.html
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_images_labels(list_files):
    nFiles = len(list_files)
    imgs = np.zeros((0, img_size*img_size*3) ,dtype=np.uint8)
    labels = np.zeros(0, dtype=np.uint8)
    for i in range(nFiles):
        dict_imgs_labels = unpickle(list_files[i])
        imgs = np.concatenate([imgs, dict_imgs_labels[b'data']], 0)
        labels = np.concatenate([labels, dict_imgs_labels[b'labels']], 0)
    imgs  = np.resize(imgs, (imgs.shape[0], 3, img_size, img_size))
    dict_label_names = unpickle(lsit_files_label_names[0])
    label_names = dict_label_names[b'label_names']
    return imgs, labels, label_names