import numpy as np
import pandas as pd

from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils.io_utils import HDF5Matrix
from os.path import join
from PIL import Image
from sys import argv


def run_preprocessing(image_dir):
    # resize and preprocess images
    h5_name = join(
            image_dir,
            'food_c101_n1000_r384x384x3.h5')
    image_store = HDF5Matrix(
            h5_name,
            'images')
    images = np.array(image_store)
    image_list = []
    for ix in range(images.shape[0]):
        image = images[ix, :, :, :]
        image = Image.fromarray(image)
        image = image.resize((224, 224), Image.BICUBIC)
        image_list.append(image)
    images = np.array(
            [np.array(image) for image in image_list],
            dtype=np.float64)
    images = preprocess_input(images)
    # handle categories
    category_store = HDF5Matrix(
            h5_name,
            '/category')
    categories = np.array(category_store)
    category_name_store = HDF5Matrix(
            h5_name,
            '/category_names')
    category_names = np.array(category_name_store)
    labels_hr = [
            category_names[ix].decode('UTF-8')
            for ix in np.where(categories)[1]]
    # return images and human-readable labels
    return images, labels_hr


if __name__ == '__main__':
    image_dir = argv[1]
    images, labels = run_preprocessing(image_dir)
    median_df = pd.read_csv('median_caloric_density.csv')
    model = VGG16(include_top=False, weights='imagenet')
    features = model.predict(images)
    np.save('features.npy', features)

