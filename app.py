from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
import tensorflow
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array



# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

IMAGE_SIZE = (150, 150)


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()


#Load your trained model
model = tensorflow.keras.models.load_model('model.h5',compile=False)

modell=model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#model._make_predict_function()          # Necessary to make everything ready to run on the GPU ahead of time
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#print('Model loaded. Check http://127.0.0.1:5000/')


def image_preprocessor(path):
    '''
    Function to pre-process the image before feeding to model.
    '''
    print('Processing Image ...')
    currImg_BGR = cv2.imread(path)
    b, g, r = cv2.split(currImg_BGR)
    currImg_RGB = cv2.merge([r, g, b])
    currImg = cv2.resize(currImg_RGB, IMAGE_SIZE)
    currImg = currImg/255.0
    currImg = np.reshape(currImg, (1, 150, 150, 3))
    return currImg

def model_predict(image):
    '''
    Perfroms predictions based on input image
    '''
    print("Image_shape", image.shape)
    print("Image_dimension", image.ndim)
    # Returns Probability:
    # prediction = model.predict(image)[0]
    # Returns class:
    prediction = model.predict_classes(image)[0]
    '''    if prediction == 1:
        return "Pneumonia"
    else:
        return "Normal"'''
    return (prediction)


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

        #image_preprocessing
        image = image_preprocessor(file_path)

        # Make prediction
        preds = model_predict(image)
        os.remove(file_path)#removes file from the server after prediction has been returned

        # Arrange the correct return according to the model. 
		# In this model 1 is Pneumonia and 0 is Normal.
        str1 = 'Pneumonia'
        str2 = 'Normal'
        if preds == 1:
            return str1
        else:
            return str2
    return None

    #this section is used by gunicorn to serve the app on Heroku
if __name__ == '__main__':
        app.run(debug=True)
    #uncomment this section to serve the app locally with gevent at:  http://localhost:5000
    # Serve the app with gevent 
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()
