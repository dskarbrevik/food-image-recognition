import os
import numpy as np
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import matplotlib.pyplot as plt
import matplotlib.image as img

import h5py
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model
from urllib import urlopen

from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.layers import Input
import imageAugmenter as T

import utils
import file_utils
import model_utils

from utils import load_train
from utils import load_test



print 'Starting to Load Training Data'
X_train, y_train = load_train()
print 'Done Loading Training Data'
print 'Starting to Load Test Data'
X_test, y_test = load_test()
print 'Done Loading Test Data'

n_classes = 1
y_train_cat = to_categorical(y_train, nb_classes=n_classes)
y_test_cat = to_categorical(y_test, nb_classes=n_classes)

import multiprocessing as mp
num_processes = 6
#pool = mp.Pool(processes=num_processes)

train_datagen = T.ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False, # randomly flip images
    zoom_range=[.8, 1],
    channel_shift_range=30,
    fill_mode='reflect')
train_datagen.config['random_crop_size'] = (299, 299)
train_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
train_generator = train_datagen.flow(X_train, y_train_cat, batch_size=64, seed=11)

test_datagen = T.ImageDataGenerator()
test_datagen.config['random_crop_size'] = (299, 299)
test_datagen.set_pipeline([T.random_transform, T.random_crop, T.preprocess_input])
test_generator = test_datagen.flow(X_test, y_test_cat, batch_size=64, seed=11)

## Baseline Model
from utils import predictImage
from utils import load_model

#baseline_model = load_model('/Users/amitabha/w210-food-image-recognition/mobile/iOS/model4b.10-0.68.hdf5')
#predictImage('http://food.fnr.sndimg.com/content/dam/images/food/fullset/2007/7/13/0/PXSP01_Baklava.jpg.rend.hgtvcom.616.462.suffix/1384784356294.jpeg')

## Resnet Model
from model_utils import createResNetModel
from model_utils import compileAndFitModel

resnet_model = createResNetModel()

compileAndFitModel(resnet_model, train_generator, test_generator, X_train, X_test, y_train, y_test)

preds_top_1 = {k: collections.Counter(v[0]).most_common(1) for k, v in preds_with_crop.items()}
preds_10_crop = {}
for ix in range(len(X_test)):
    if ix % 1000 == 0:
        print(ix)
    preds_10_crop[ix] = preds_with_crop(X_test[ix], ix)
    

top_5_per_ix = {k: collections.Counter(preds_with_crop[k][1].reshape(-1)).most_common(5) 
                for k, v in preds_10_crop.items()}
preds_top_5 = {k: [y[0] for y in v] for k, v in top_5_per_ix.items()}
y_pred = [x[0][0] for x in preds_top_1.values()]

from model_utils import plot_confusion_matrix

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

class_names = [ix_to_class[i] for i in range(101)]

plt.figure()
fig = plt.gcf()
fig.set_size_inches(32, 32)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization',
                      cmap=plt.cm.cool)
plt.show()

