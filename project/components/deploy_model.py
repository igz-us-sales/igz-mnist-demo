import os
import sys
import numpy as np
import json
import base64
from io import BytesIO

import mlrun
from mlrun.artifacts import get_model
from PIL import Image
from tensorflow.keras.models import load_model

class MNISTModel(mlrun.serving.V2ModelServer):
    def load(self):
        """load and initialize the model and/or other elements"""
        model_file, extra_data = self.get_model('.h5')
        self.model = load_model(model_file)
        
    def decode_image(self, byte_stream):
        """Decode image from base64 encoded string"""
        im_bytes = base64.b64decode(byte_stream)
        im_file = BytesIO(im_bytes)
        img = Image.open(im_file)
        return img
    
    def predict_digit(self, img):
        """Get model prediction and accuracy"""
        #resize image to 28x28 pixels
        img = img.resize((28,28))
        #convert rgb to grayscale
        img = img.convert('L')
        img = np.array(img)
        #reshaping to support our model input and normalizing
        img = img.reshape(1,28,28,1)
        img = img/255.0
        #predicting the class
        res = self.model.predict([img])[0]
        return np.argmax(res), max(res) 

    def predict(self, body):
        """Generate model predictions from sample."""
        img = np.asarray(body['inputs'])
        digit, acc = self.predict_digit(self.decode_image(img))
        return int(digit), float(acc)

from mlrun.runtimes import nuclio_init_hook
def init_context(context):
    nuclio_init_hook(context, globals(), 'serving_v2')

def handler(context, event):
    return context.mlrun_handler(context, event)
