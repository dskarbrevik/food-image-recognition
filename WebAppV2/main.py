# from flask import \
#         Flask, \
#         redirect, \
#         render_template, \
#         request, \
#         send_from_directory, \
#         url_for
# import sys
# import logging
# import os
# import re
# from google.cloud import storage
# from werkzeug.utils import secure_filename
# # from google.appengine.api import users
# # from google.appengine.ext import blobstore
# # from google.appengine.ext import ndb
# # from google.appengine.ext.webapp import blobstore_handlers
#
# # ALLOWED_EXTENSIONS = set([
# #     'txt',
# #     'pdf',
# #     'png',
# #     'jpg',
# #     'jpeg',
# #     'gif'])
#
# app = Flask(__name__)
#
#
# CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']
#
#
# # def allowed_file(filename):
# #     return '.' in filename and \
# #     filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     # if request.method == 'POST':
#     #     test_pred = "test3"
#     #     file = request.files['file']
#     #     if file.filename == '':
#     #         flash('No selected image')
#     #         return redirect(request.url)
#     #     if file and allowed_file(file.filename):
#     #         filename = secure_filename(file.filename)
#     #
#     #
#     #     return render_template("prediction.html", test_pred=test_pred)
#     # else:
#
#     #uploadURI = blobstore.create_upload_url('/submit', gs_bucket_name="foodhud-project.appspot.com")
#
#     test="test2"
#     return render_template("index.html", test=test)
#
# @app.route('/upload', methods=['POST'])
# def upload():
#     """Process the uploaded file and upload it to Google Cloud Storage."""
#     uploaded_file = request.files.get('file')
#     #
#     # if not uploaded_file:
#     #     return 'No file uploaded.', 400
#
#     # Create a Cloud Storage client.
#     gcs = storage.Client()
#
#     # Get the bucket that the file will be uploaded to.
#     bucket = gcs.get_bucket(CLOUD_STORAGE_BUCKET)
#
#     # Create a new blob and upload the file's content.
#     blob = bucket.blob(uploaded_file.filename)
#
#     blob.upload_from_string(
#         uploaded_file.read(),
#         content_type=uploaded_file.content_type
#     )
#
#     # The public URL can be used to directly access the uploaded file via HTTP.
#     return render_template("prediction.html", bloburl=blob.public_url)
#
# # @app.errorhandler(500)
# # def server_error(e):
# #     logging.exception('An error occurred during a request.')
# #     return """
# #     An internal error occurred: <pre>{}</pre>
# #     See logs for full stacktrace.
# #     """.format(e), 500
#
# # @app.route("/submit", methods=['POST'])
# # def submit():
# #     if request.method == 'POST':
# #         f = request.files['file']
# #         header = f.headers['Content-Type']
# #         parsed_header = parse_options_header(header)
# #         blob_key = parsed_header[1]['blob-key']
# #         return render_template("prediction.html", bkey=blob_key)
#
#
# # @app.route("/img/<bkey>")
# # def img(bkey):
# #     blob_info = blobstore.get(bkey)
# #     response = make_response(blob_info.open().read())
# #     response.headers['Content-Type'] = blob_info.content_type
# #     return response
#
# if __name__ == '__main__':
#     # This is used when running locally. Gunicorn is used to run the
#     # application on Google App Engine. See entrypoint in app.yaml.
#     app.run(host='127.0.0.1', port=8080, debug=True)



# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START app]
import logging
import os
import re
#from PIL import Image
#from io import BytesIO
import numpy as np
from scipy.misc import imsave, imread, imresize

from flask import Flask, request, render_template
from google.cloud import storage

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import \
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
    uploadURL = blob.public_url
    with open('/tmp/my-weird-blob', 'wb') as blob_obj:
        blob.download_to_file(blob_obj)
    #image_str = BytesIO()
    #image_str.write(blob_obj.read())
    #image_str.seek(0)
    # image_ = Image.open('blob_obj')
    # image = image_.resize((224, 224))
    # # preprocess the image
    # # keeping only color channels
    # x = np.array(image, dtype=np.float64)
    # x = np.expand_dims(x, axis=0)
    # x = x[:, :, :, :3]
    # x = preprocess_input(x)

        img = image.load_img('/tmp/my-weird-blob', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        #blob_image = blob_to_image(blob_obj)
        #processed_image = blob_to_image(blob_image)
        #preds = make_prediction(processed_image)
        model = ResNet50(weights='imagenet')
        preds = model.predict(image_x)
        predictions = decode_predictions(
                preds,
                top=3)[0]
        top_1 = predictions[0][1]
        top_2 = predictions[1][1]
        top_3 = predictions[2][1]
        # top1 = preds[0]
        # top2 = preds[1]
        # top3 = preds[2]
    # The public URL can be used to directly access the uploaded file via HTTP.
        return render_template("prediction.html", bloburl=uploadURL, top1=top_1, top2=top_2, top3=top_3)


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
