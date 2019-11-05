# ---------------------------
# Matthew Willetts, Alexander Camuto -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: mwilletts@turing.ac.uk, acamuto@turing.ac.uk
# ---------------------------
"""Functions to preprocess cifar10 data
"""
import numpy as np
import tensorflow as tf

import os
import sys
import bz2
import urllib.request

URL = ['https://www.dropbox.com/s/nflroijnpeqn3b2/cifar10_embeddings.npz.bz2?dl=1']


def load_cifar10(normalize=True, dequantify=True):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
    :return:
    '''
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()

    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_y = tf.keras.utils.to_categorical(train_y.astype('int32'), 10)
    test_y = tf.keras.utils.to_categorical(test_y.astype('int32'), 10)

    if dequantify:
        train_x += np.random.uniform(0,1,size=train_x.shape).astype('float32')
        test_x += np.random.uniform(0,1,size=test_x.shape).astype('float32')

    if normalize:
        normalizer = train_x.max().astype('float32')
        train_x = train_x / normalizer
        test_x = test_x / normalizer

    return train_x, train_y, test_x, test_y


# Embedding loading code

def unzip(directory, filename):
    npz_file_path = os.path.join(directory,filename)
    bz2_file_path = os.path.join(directory,filename+'.bz2')
    if os.path.isfile(npz_file_path) == False:
        print("Extracting")
        with open(npz_file_path, 'wb') as new_file, bz2.BZ2File(bz2_file_path, 'rb') as file:
            for data in iter(lambda : file.read(100 * 1024), b''):
                new_file.write(data)


def download(directory, filename, url):
    """Downloads file"""
    filepath = os.path.join(directory, filename)
    if tf.gfile.Exists(filepath):
        return filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    print("Downloading %s to %s" % (url, filepath))
    urllib.request.urlretrieve(url, filepath)
    return filepath


def load_cifar10_embeddings(directory, URL, file='cifar10_embeddings.npz'):
    bz2_file = file+'.bz2'
    npz_file_path = os.path.join(directory, file)
    download(directory, bz2_file, URL[0])
    unzip(directory, file)
    data_loaded = np.load(npz_file_path)
    x_train, y_train, x_test, y_test = data_loaded['x_train'], data_loaded['y_train'], data_loaded['x_test'], data_loaded['y_test']
    return x_train, y_train, x_test, y_test