# ---------------------------
# Matthew Willetts, Alexander Camuto -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: mwilletts@turing.ac.uk, acamuto@turing.ac.uk
# ---------------------------
"""Script to load up resnet-50 embeddings of cifar-10 and svhn
"""

import numpy as np

from svhn import load_svhn_embeddings
from cifar10 import load_cifar10_embeddings


# path to where you have downloaded the npz.bz2 files
# OR where you want them to be downloaded to, for SVHN and CIFAR-10
cifar_embeddings_path = './Data/cifar'
svhn_embeddings_path = './Data/svhn'

CIFAR_URL = ['https://www.dropbox.com/s/nflroijnpeqn3b2/cifar10_embeddings.npz.bz2?dl=1']
SVHN_URL = ['https://www.dropbox.com/s/o5ke5h5froi0dvy/svhn_embeddings.npz.bz2?dl=1']


x_train, y_train, x_test, y_test = load_svhn_embeddings(svhn_embeddings_path, SVHN_URL)

svhn_dict = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}

x_train, y_train, x_test, y_test = load_cifar10_embeddings(cifar_embeddings_path, CIFAR_URL)

cifar_dict = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
