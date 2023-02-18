from __future__ import division, print_function


import os

import numpy as np


# Keras

from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.models import load_model

# from keras.preprocessing import image


import keras.utils as image


# Flask utils

from flask import Flask, redirect, url_for, request, render_template

from werkzeug.utils import secure_filename


# Define a flask app

app = Flask(__name__)


# Model saved with Keras model.save()

MODEL_PATH = 'binary_trained_final.h5'


# Load your trained model

model = load_model(MODEL_PATH)

model.make_predict_function()  # Necessary


print('Model loaded.')


def model_predict(img_path, model):

    img = image.load_img(img_path, target_size=(256, 256))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)

    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)

    return preds


@app.route('/', methods=['GET'])
def index():

    # Main page

    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():

    if request.method == 'POST':

        # Get the file from post request

        f = request.files['file']

        # Save the file to ./uploads

        basepath = os.path.dirname(__file__)

        file_path = os.path.join(

            basepath, 'uploads', secure_filename(f.filename))

        f.save(file_path)

        # Make prediction

        preds = model_predict(file_path, model)

        x = np.argmax(preds)
        print(x)

        #

        # if(preds[0][0]==1):

        #     return "i"

        # else:

        #     return 'h'

        if (x == 1):

            return "Infected"

        else:

            return "Healthy"

    return None


# if __name__ == '__main__':

#     app.run(debug=True)
