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

def copytree(src, dst, symlinks = False, ignore = None):
  if not os.path.exists(dst):
      os.makedirs(dst)
      shutil.copystat(src, dst)
  lst = os.listdir(src)
  if ignore:
      excl = ignore(src, lst)
      lst = [x for x in lst if x not in excl]
  for item in lst:
      s = os.path.join(src, item)
      d = os.path.join(dst, item)
      if symlinks and os.path.islink(s):
          if os.path.lexists(d):
              os.remove(d)
          os.symlink(os.readlink(s), d)
          try:
              st = os.lstat(s)
              mode = stat.S_IMODE(st.st_mode)
              os.lchmod(d, mode)
          except:
              pass # lchmod not available
      elif os.path.isdir(s):
          copytree(s, d, symlinks, ignore)
      else:
          shutil.copy2(s, d)



def generate_dir_file_map(path):
  dir_files = defaultdict(list)
  with open(path, 'r') as txt:
      files = [l.strip() for l in txt.readlines()]
      for f in files:
          dir_name, id = f.split('/')
          dir_files[dir_name].append(id + '.jpg')
  return dir_files


def ignore_train(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = train_dir_files[subdir]
        return to_ignore


def ignore_test(d, filenames):
        print(d)
        subdir = d.split('/')[-1]
        to_ignore = test_dir_files[subdir]
        return to_ignore

def splitFiles():
  if not os.path.isdir('./food-101/test') and not os.path.isdir('./food-101/train'):
    train_dir_files = generate_dir_file_map('food-101/meta/train.txt')
    test_dir_files = generate_dir_file_map('food-101/meta/test.txt')
    copytree('food-101/images', 'food-101/test', ignore=ignore_train)
    copytree('food-101/images', 'food-101/train', ignore=ignore_test)




