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
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.utils.np_utils import to_categorical
from keras.optimizers import SGD
from keras.regularizers import l2
import keras.backend as K
import math

from keras.preprocessing import image
from keras.layers import Input

#InceptionV3
from keras.applications.inception_v3 import InceptionV3

#ResNet50
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


def createResNetModel():
  K.clear_session()
  #bn_axis = 3
  #img_rows, img_cols, color_type  = 229, 229, 1 # Resolution of inputs
  #channel = 3
  #n_classes = 1
  #batch_size = 16
  #nb_epoch = 10
  #img_input = Input(shape=(img_rows, img_cols, color_type), name='data')
  
  #x = resnet_model.output
  #x = AveragePooling2D(pool_size=(7, 7))(x)
  #x = Dropout(.4)(x)
  #x = Flatten()(x)
  #x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
  #x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1', bias=False)(x)
  #x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
  #x = Scale(axis=bn_axis, name='scale_conv1')(x)
  #x = Activation('relu', name='conv1_relu')(x)
  #x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
  
  #x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
  #x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
  #x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
  
  #x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
  #for i in range(1,4):
  #  x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i))
  
  #x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
  #for i in range(1,23):
  #  x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i))
  
  #x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
  #x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
  #x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
  
  #x_fc = AveragePooling2D((7, 7), name='avg_pool')(x)
  #x_fc = Flatten()(x_fc)
  #x_fc = Dense(1000, activation='softmax', name='fc1000')(x_fc)
 
  #resnet_model = ResNet50(weights='imagenet', include_top=True, img_input, x_fc)
  n_classes=101 
  
  #predictions = Dense(n_classes, W_regularizer=l2(.0005), activation='softmax')(x)
  #resnet_model = Model(input=resnet_model.input, output=predictions)
  #sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
  #model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

  base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(299, 299, 3)))
  x = base_model.output
  x = AveragePooling2D(pool_size=(1, 1))(x)
  x = Dropout(.4)(x)
  x = Flatten()(x)
  predictions = Dense(n_classes, init='glorot_uniform', W_regularizer=l2(.0005), activation='softmax')(x)

  model = Model(input=base_model.input, output=predictions)
  sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  return model

def schedule(epoch):
    if epoch < 15:
        return .01
    elif epoch < 28:
        return .002
    else:
        return .0004


def compileAndFitModel(model, train_generator, test_generator, X_train, X_test, y_train, y_test):
  n_classes = 1 
  print 'From complieAndFit y_train.shape[0] :: ', y_train.shape[0]
  print 'From complieAndFit y_test.shape[0] :: ', y_test.shape[0]
  checkpointer = ModelCheckpoint(filepath='resnet.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_best_only=True)
  csv_logger = CSVLogger('resnet.log')
  lr_scheduler = LearningRateScheduler(schedule)  
  print 'From complieAndFit X_test.shape[0] :: ', X_test.shape[0]
  print 'From complieAndFit X_train.shape[0] :: ', X_train.shape[0]
  model.fit_generator(train_generator,
                    validation_data=test_generator,
                    nb_val_samples=X_test.shape[0],
                    samples_per_epoch=X_train.shape[0],
                    nb_epoch=32,
  		    verbose=2)
                    #, callbacks = [lr_scheduler, csv_logger, checkpointer])
  #model.fit(X_train, y_train,
  #            shuffle=True,
  #            verbose=1,
  #            callbacks = [csv_logger, checkpointer], 
  #            validation_data=(X_test, y_test)
  #            )

		  

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
