# ---------------------------
# Matthew Willetts, Alexander Camuto -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: mwilletts@turing.ac.uk, acamuto@turing.ac.uk
# ---------------------------
"""Functions to preprocess SVHN data
"""
import numpy as np
import tensorflow as tf

import os
import sys
import shutil
import zipfile
import scipy.misc
import scipy.io as sio
import pickle as Pkl
import gzip, tarfile
import re, string, fnmatch
import urllib.request
import bz2

URL = ['https://www.dropbox.com/s/o5ke5h5froi0dvy/svhn_embeddings.npz.bz2?dl=1']


def _get_datafolder_path():
    full_path = os.path.abspath('.')
    path = full_path +'/data'
    return path


def load_svhn(
        dataset=_get_datafolder_path()+'/svhn/',
        normalize=True,
        dequantify=True,
        extra=False):
    '''
    :param dataset:
    :param normalize:
    :param dequantify: Add uniform noise to dequantify the data following
        Uria et. al 2013
        "RNADE: The real-valued neural autoregressive density-estimator"
    :param extra: include extra svhn samples
    :return:
    '''

    if not os.path.isfile(dataset +'svhn_train.pkl'):
        datasetfolder = os.path.dirname(dataset +'svhn_train.pkl')
        if not os.path.exists(datasetfolder):
            os.makedirs(datasetfolder)
        _download_svhn(dataset, extra=False)

    with open(dataset +'svhn_train.pkl', 'rb') as f:
        train_x,train_y = Pkl.load(f)
    with open(dataset +'svhn_test.pkl', 'rb') as f:
        test_x,test_y = Pkl.load(f)

    if extra:
        if not os.path.isfile(dataset +'svhn_extra.pkl'):
            datasetfolder = os.path.dirname(dataset +'svhn_train.pkl')
            if not os.path.exists(datasetfolder):
                os.makedirs(datasetfolder)
            _download_svhn(dataset, extra=True)

        with open(dataset +'svhn_extra.pkl', 'rb') as f:
            extra_x,extra_y = Pkl.load(f)
        train_x = np.concatenate([train_x,extra_x])
        train_y = np.concatenate([train_y,extra_y])

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



def _download_svhn(dataset, extra):
    """
    Download the SVHN dataset
    """
    from scipy.io import loadmat

    print('Downloading data from http://ufldl.stanford.edu/housenumbers/, ' \
          'this may take a while...')
    if extra:
        print("Downloading extra data...")
        urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat',
                           dataset+'extra_32x32.mat')
        extra = loadmat(dataset+'extra_32x32.mat')
        extra_x = extra['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
        extra_y = extra['y'].reshape((-1)) - 1

        print("Saving extra data")
        with open(dataset +'svhn_extra.pkl', 'wb') as f:
            Pkl.dump([extra_x,extra_y],f,protocol=Pkl.HIGHEST_PROTOCOL)
        os.remove(dataset+'extra_32x32.mat')

    else:
        print("Downloading train data...")
        urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
                           dataset+'train_32x32.mat')
        print("Downloading test data...")
        urllib.request.urlretrieve('http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                           dataset+'test_32x32.mat')

        train = loadmat(dataset+'train_32x32.mat')
        train_x = train['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
        train_y = train['y'].reshape((-1)) - 1
        test = loadmat(dataset+'test_32x32.mat')
        test_x = test['X'].swapaxes(2,3).swapaxes(1,2).swapaxes(0,1)
        test_y = test['y'].reshape((-1)) - 1

        print("Saving train data")
        with open(dataset +'svhn_train.pkl', 'wb') as f:
            Pkl.dump([train_x,train_y],f,protocol=Pkl.HIGHEST_PROTOCOL)
        print("Saving test data")
        with open(dataset +'svhn_test.pkl', 'wb') as f:
            Pkl.dump([test_x,test_y],f,protocol=Pkl.HIGHEST_PROTOCOL)
        os.remove(dataset+'train_32x32.mat')
        os.remove(dataset+'test_32x32.mat')


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


def load_svhn_embeddings(directory, URL, file='svhn_embeddings.npz'):
    bz2_file = file+'.bz2'
    npz_file_path = os.path.join(directory, file)
    download(directory, bz2_file, URL[0])
    unzip(directory, file)
    data_loaded = np.load(npz_file_path)
    x_train, y_train, x_test, y_test = data_loaded['x_train'], data_loaded['y_train'], data_loaded['x_test'], data_loaded['y_test']
    return x_train, y_train, x_test, y_test