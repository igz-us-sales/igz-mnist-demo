from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import numpy as np

def build_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])
    return model
    
def handler(context):
    # Get X, y
    X_train = np.load(str(context.get_input("X_train")), allow_pickle=True)
    y_train = np.load(str(context.get_input("y_train")), allow_pickle=True)
    X_test = np.load(str(context.get_input("X_test")), allow_pickle=True)
    y_test = np.load(str(context.get_input("y_test")), allow_pickle=True)
    
    # Get training params
    batch_size = context.get_param("batch_size")
    num_classes = context.get_param("num_classes")
    epochs = context.get_param("epochs")
    
    # Build model
    model = build_model(num_classes)
    
    # Define training callbacks
    tensorboard_callback = TensorBoard(log_dir=f"/User/tensorboard/{context.name}-{context.uid}")
    callbacks = [tensorboard_callback]

    # Train model
    hist = model.fit(X_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=epochs,
                     verbose=1,
                     validation_data=(X_test, y_test),
                     callbacks=callbacks)
    context.logger.info("The model has successfully trained")

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    context.logger.info(f"Test loss: {test_loss}")
    context.logger.info(f"Test accuracy: {test_accuracy}")

    # Log model and metrics
    model.save('mnist.h5')
    context.logger.info("Saving the model as mnist.h5")
    
    context.log_results({"test_loss" : test_loss,
                         "test_accuracy" : test_accuracy})
    
    context.log_model("model",
                      artifact_path=context.artifact_path,
                      model_file="mnist.h5",
                      metrics=context.results)