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

def load_images(root, min_side=299):
    all_imgs = []
    all_classes = []
    resize_count = 0
    invalid_count = 0
    class_to_ix = {}
    ix_to_class = {}
    with open('food-101/meta/classes.txt', 'r') as txt:
      classes = [l.strip() for l in txt.readlines()]
      class_to_ix = dict(zip(classes, range(len(classes))))
      ix_to_class = dict(zip(range(len(classes)), classes))
      class_to_ix = {v: k for k, v in ix_to_class.items()}
    sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))
    for i, subdir in enumerate(listdir(root)):
        imgs = listdir(join(root, subdir))
        class_ix = class_to_ix[subdir]
        print(i, class_ix, subdir)
        for img_name in imgs:
            img_arr = img.imread(join(root, subdir, img_name))
            img_arr_rs = img_arr
            try:
                w, h, _ = img_arr.shape
                if w < min_side:
                    wpercent = (min_side/float(w))
                    hsize = int((float(h)*float(wpercent)))
                    img_arr_rs = imresize(img_arr, (min_side, hsize))
                    resize_count += 1
                elif h < min_side:
                    hpercent = (min_side/float(h))
                    wsize = int((float(w)*float(hpercent)))
                    img_arr_rs = imresize(img_arr, (wsize, min_side))
                    resize_count += 1
                all_imgs.append(img_arr_rs)
                all_classes.append(class_ix)
            except:
                print('Skipping bad image: ', subdir, img_name)
                invalid_count += 1
    print(len(all_imgs), 'images loaded')
    print(resize_count, 'images resized')
    print(invalid_count, 'images skipped')
    return np.array(all_imgs), np.array(all_classes)

def predict_with_crops(img, ix, model, top_n=5, preprocess=True):
    flipped_X = np.fliplr(img)
    crops = [
        img[:299,:299, :], # Upper Left
        img[:299, img.shape[1]-299:, :], # Upper Right
        img[img.shape[0]-299:, :299, :], # Lower Left
        img[img.shape[0]-299:, img.shape[1]-299:, :], # Lower Right
        center_crop(img, (299, 299)),

    flipped_X[:299,:299, :],
        flipped_X[:299, flipped_X.shape[1]-299:, :],
        flipped_X[flipped_X.shape[0]-299:, :299, :],
        flipped_X[flipped_X.shape[0]-299:, flipped_X.shape[1]-299:, :],
        center_crop(flipped_X, (299, 299))
    ]
    if preprocess:
        crops = [preprocess_input(x.astype('float32')) for x in crops]


    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    top_5_preds= np.argpartition(y_pred, -top_n)[:,-top_n:]
    return preds, top_5_preds

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0]//2, x.shape[1]//2
    halfw, halfh = center_crop_size[0]//2, center_crop_size[1]//2
    return x[centerw-halfw:centerw+halfw+1,centerh-halfh:centerh+halfh+1, :]

def load_test():
  return load_images('./food-101/test', min_side=299)

def load_train():
  return load_images('food-101/train', min_side=299)

def predictImage(url, model):
    f = urlopen(url)
    pic = plt.imread(f, format='jpg')
    preds = predict_with_crops(np.array(pic), 0, model)[0]
    best_pred = collections.Counter(preds).most_common(1)[0][0]
    print(ix_to_class[best_pred])
    plt.imshow(pic)
