# [START app]
import logging
import os
import re
from PIL import Image
from io import BytesIO
import numpy as np
from scipy.misc import imsave, imread, imresize

from flask import Flask, request, render_template
from google.cloud import storage
import requests
from keras.models import load_model
from keras.preprocessing import image
# from keras.applications.resnet50 import \
#         decode_predictions, \
#         preprocess_input
from keras.applications.inception_v3 import \
        decode_predictions, \
        preprocess_input

# [start config]
app = Flask(__name__)


# Configure this environment variable via app.yaml
CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
# [end config]


# def blob_to_image(blob_obj):
#     image_str = BytesIO()
#     image_str.write(blob_obj.read())
#     image_str.seek(0)
#
#     image_ = Image.open(image_str)
#     image = image_.resize((224, 224))
#     # preprocess the image
#     # keeping only color channels
#     x = np.array(image, dtype=np.float64)
#     x = np.expand_dims(x, axis=0)
#     x = x[:, :, :, :3]
#     x = preprocess_input(x)
#     return x
#
# def make_prediction(image_x):
#     model = ResNet50(weights='imagenet')
#     preds = model.predict(image_x)
#     predictions = decode_predictions(
#             preds,
#             top=3)[0]
#     top_1 = predictions[0][1]
#     top_2 = predictions[1][1]
#     top_3 = predictions[2][1]
#     preds_list = [top_1, top_2, top_3]
#     return preds_list

##################################
## load the model in first ######
##################################
gcs = storage.Client()
bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
blob_model = bucket.blob("model4b.10-0.68.hdf5")
blob_label = bucket.blob("labels.npy")
model_file = blob_model.download_to_filename("model.hdf5")
model_labels = blob_label.download_to_filename("labels.npy")
#model = load_model("resnet_model.h5")
model = load_model("model.hdf5")
labels = np.load("labels.npy")
######################################
######################################

# [START form]
@app.route('/')
def index():
    return render_template("index.html")

# [START upload]
@app.route('/upload', methods=['POST'])
def upload():
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return 'No file uploaded.', 400

    ### Upload image to google cloud ######
    # Create a Cloud Storage client.
    gcs = storage.Client()
    # Get the bucket that the file will be uploaded to.
    bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
    bucket.make_public(recursive=True, future=True)
    # Create a new blob and upload the file's content.
    blob = bucket.blob(uploaded_file.filename)
    blob.upload_from_string(
        uploaded_file.read(),
        content_type=uploaded_file.content_type
    )

    ### Download image from google cloud #####
    uploadURL = blob.public_url # address to view image
    response = requests.get(uploadURL)
    img = Image.open(BytesIO(response.content))


    ## pre-process image and prediction for resnet50 model #####
    # image = img.resize((224, 224))
    # x = np.array(image, dtype=np.float64)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # preds = model.predict(x)
    # predictions = decode_predictions(preds, top=3)[0]
    # top_1 = predictions[0][1]
    # top_2 = predictions[1][1]
    # top_3 = predictions[2][1]


    k = 5
    image = img.resize((299, 299), resample=Image.BICUBIC)
    x = np.array(image, dtype='float64')
    x = preprocess_input(x)
    x = np.expand_dims(x, axis=0)
    x = x[:, :, :, :3]
    predictions = model.predict(x)
    descending_ix = np.argsort(-1*predictions.flatten())
    top_k_labels = labels[descending_ix][:k]
    top_1 = str(top_k_labels[0])

    return render_template("prediction.html", uploadURL=uploadURL, top_1=top_1)
    #return render_template("prediction.html", uploadURL=uploadURL, top_1=top_1, top_2=top_2, top_3=top_3)

        # top1 = preds[0]
        # top2 = preds[1]
        # top3 = preds[2]

# [END upload]


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [END app]
