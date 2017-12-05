import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils.io_utils import HDF5Matrix
from os.path import join
from PIL import Image
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_squared_error
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


def get_cd_labels(labels_hr, median_df):
    # compute array with caloric density
    # entries corresponding to foods in
    # label set
    cd_lookup = dict()
    for index, row in median_df.iterrows():
        cd_lookup[row['label']] = row['caloric_density']
    labels_n = []
    for label in labels_hr:
        n = np.nan \
                if label not in cd_lookup \
                else cd_lookup[label]
        labels_n.append(n)
    labels_n = np.array(labels_n)
    return labels_n


def compute_cv_scores(
        model,
        features,
        labels_n,
        n_cv=5,
        n_comp_max=100):
    ks = list(range(1, n_comp_max+1))
    cv_scores = []
    valid_ix = ~np.isnan(labels_n)
    for k in ks:
        pca = PCA(n_components=k)
        pca_features = pca.fit_transform(features)
        cv_score = cross_val_score(
                model,
                pca_features[valid_ix, :],
                labels_n[valid_ix],
                scoring=make_scorer(mean_squared_error),
                cv=n_cv).mean()
        cv_scores.append(cv_score)
    return ks, cv_scores


if __name__ == '__main__':
    image_dir = argv[1]
    images, labels_hr = run_preprocessing(image_dir)
    median_df = pd.read_csv('median_caloric_density.csv')
    labels_n = get_cd_labels(labels_hr, median_df)
    featurizer = VGG16(include_top=False, weights='imagenet')
    features = featurizer.predict(images)
    features = features.reshape(
            features.shape[0],
            -1)
    np.save('features.npy', features)
    np.save('labels.npy', labels_n)
    model = LinearRegression()
    ks, cv_scores = compute_cv_scores(
            model,
            features,
            labels_n)
    plt.plot(ks, cv_scores)
    plt.xlabel('Principal Components')
    plt.ylabel('Mean Squared Error')
    plt.title('Linear Model with VGG16-Derived Features')
    plt.show()
