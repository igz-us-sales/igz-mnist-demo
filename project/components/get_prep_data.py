from tensorflow import keras
from tensorflow.keras.datasets import mnist
import numpy as np
import os

def handler(context):
    # the data, split between train and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    context.logger.info(f'X_train shape: {X_train.shape}')
    context.logger.info(f'y_train shape: {y_train.shape}')

    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    context.logger.info(f'X_train shape: {X_train.shape}')
    context.logger.info(f'train samples: {X_train.shape[0]}')
    context.logger.info(f'test samples: {X_test.shape[0]}')

    # Create output dir
    os.makedirs(context.artifact_path, exist_ok=True)
    
    # Log X, y to MLRun DB
    output_data = {"X_train" : X_train,
                   "X_test" : X_test,
                   "y_train" : y_train,
                   "y_test" : y_test}
    for key, value in output_data.items():
        np.save(f"{context.artifact_path}/{key}", value)
        context.log_result(key, f"{context.artifact_path}/{key}.npy")