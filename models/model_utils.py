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

from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math

from keras.preprocessing import image
from keras.layers import Input

#ResNet50
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


def createResNetModel():
  K.clear_session()
  resnet_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
  x = resnet_model.output
  x = AveragePooling2D(pool_size=(8, 8))(x)
  x = Dropout(.4)(x)
  x = Flatten()(x)
  predictions = Dense(n_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)
  resnet_model = Model(input=resnet_model.input, output=predictions)
  return resnet_model

def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004


def compileAndFitModel(model, name, train_generator, X_train):
  opt = SGD(lr=.01, momentum=.9)
  model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
  checkpointer = ModelCheckpoint(filepath=name.append('.{epoch:02d}-{val_loss:.2f}.hdf5'), verbose=1, save_best_only=True)
  csv_logger = CSVLogger(name.append('.log'))
  lr_scheduler = LearningRateScheduler(schedule)
  model.fit_generator(train_generator,
                    validation_data=test_generator,
                    nb_val_samples=X_test.shape[0],
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=32,
                    verbose=2,
                    callbacks=[lr_scheduler, csv_logger, checkpointer])


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
