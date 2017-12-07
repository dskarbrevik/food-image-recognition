from flask import \
        Flask, \
        redirect, \
        render_template, \
        request, \
        send_from_directory, \
        url_for
# from flask.ext.session import Session
from io import BytesIO
from PIL import Image
# for regular expressions, saves time dealing with string data
import re
# scientific computing library for saving, reading, and resizing images
from scipy.misc import imsave, imread, imresize
# system level operations (like loading files)
import sys
# for reading operating system data
import os
from werkzeug.utils import secure_filename
# from load import *

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import \
        decode_predictions, \
        preprocess_input
import numpy as np


UPLOAD_FOLDER = "{}/uploads".format(os.getcwd())
ALLOWED_EXTENSIONS = set([
    'txt',
    'pdf',
    'png',
    'jpg',
    'jpeg',
    'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    #initModel()
    #render out pre-built HTML file right on the index page
    if request.method == 'POST':
    # check if the post request has the file part
    # if 'file' not in request.files:
    #     flash('No file part')
    #     return redirect(request.url)
        file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
        if file.filename == '':
            flash('No selected image')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # use an io object to avoid saving to disk
            image_str = BytesIO()
            image_str.write(file.read())
            image_str.seek(0)
            # using pillow for reshape
            image = Image.open(image_str)
            image = image.resize((224, 224))
            # preprocess the image
            x = np.array(image, dtype=np.float64)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            # run prediction
            model = ResNet50(weights='imagenet')
            preds = model.predict(x)
            predictions = decode_predictions(
                    preds,
                    top=3)[0]
            top_1 = predictions[0][1]
            top_2 = predictions[1][1]
            top_3 = predictions[2][1]
            return render_template(
                    "prediction.html",
                    filename=filename,
                    top_1=top_1,
                    top_2=top_2,
                    top_3=top_3)

    else:
        return render_template("index.html")

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    #decide what port to run the app in
    port = int(os.environ.get('PORT', 5000))
    #run the app locally on the givn port
    app.run(host='0.0.0.0', port=port)
    #optional if we want to run in debugging mode
    #app.run(debug=True)
