import json
import numpy as np
import os
import keras
from keras.models import load_model


from azureml.core.model import Model

def init():
    global model
    # retreive the path to the model file using the model name
    model_path = Model.get_model_path('pneumonia')
    model=load_model(model_path)

def run(raw_data):
    data = np.array(json.loads(raw_data)['data'])
    data = data.astype(np.uint8)
    data = np.expand_dims(data, axis=0)
    # make prediction
    y_hat = model.predict(data)
    y_hat = round(y_hat[0][0],4)
    return json.dumps(str(y_hat))