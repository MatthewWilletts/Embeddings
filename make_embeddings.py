# ---------------------------
# Matthew Willetts, Alexander Camuto -- 2019
# The University of Oxford, The Alan Turing Institute
# contact: mwilletts@turing.ac.uk, acamuto@turing.ac.uk
# ---------------------------
"""Script to make resnet-50 embeddings of cifar-10 and svhn
"""
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.utils import to_categorical
from .svhn import load_svhn

resnet = ResNet50(include_top=False, weights='imagenet', input_shape=(32,32,3), pooling='avg')


# SVHN
x_train, y_train, x_test, y_test = load_svhn(dataset='/home/mw/Data', extra=False, normalize=False, dequantify=False)

x_train_preproc = preprocess_input(x_train.copy())
x_test_preproc = preprocess_input(x_test.copy())
print(np.max(x_train_preproc))
print(np.min(x_train_preproc))

x_train_embedd = resnet.predict(x_train_preproc)
x_test_embedd = resnet.predict(x_test_preproc)

print(np.max(x_train_embedd))
print(np.min(x_train_embedd))

np.savez('/home/mw/Data/svhn_embeddings.npz', x_train=x_train_embedd, y_train=y_train, x_test=x_test_embedd, y_test=y_test)




# CIFAR-10
(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = to_categorical(y_train.astype('int32'), 10)
y_test = to_categorical(y_test.astype('int32'), 10)

x_train_preproc = preprocess_input(x_train.copy())
x_test_preproc = preprocess_input(x_test.copy())
print(np.max(x_train_preproc))
print(np.min(x_train_preproc))

x_train_embedd = resnet.predict(x_train_preproc)
x_test_embedd = resnet.predict(x_test_preproc)

print(np.max(x_train_embedd))
print(np.min(x_train_embedd))

np.savez('/home/mw/Data/cifar10_embeddings.npz', x_train=x_train_embedd, y_train=y_train, x_test=x_test_embedd, y_test=y_test)
