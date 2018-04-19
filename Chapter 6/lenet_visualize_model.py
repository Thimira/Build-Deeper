# import the necessary packages
from keras.optimizers import SGD
from keras.utils import np_utils

# imports used to build the deep learning model
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

import numpy as np
import argparse
from keras.utils import plot_model

# Setup the argument parser to parse out command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str, default="data/lenet_weights.hdf5",
                help="(optional) Path to the weights file. Defaults to 'data/lenet_weights.hdf5'")
args = vars(ap.parse_args())


def build_lenet(width, height, depth, classes, weightsPath=None):
    # Initialize the model
    model = Sequential()

    # The first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (5, 5), padding="same",
                     input_shape=(height, width, depth)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # The second set of CONV => RELU => POOL layers
    model.add(Conv2D(50, (5, 5), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # The set of FC => RELU layers
    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))

    # The softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # If a weights path is supplied, then load the weights
    if weightsPath is not None:
        model.load_weights(weightsPath)

    # Return the constructed network architecture
    return model


# Build and Compile the model
print("[INFO] Building and compiling the LeNet model...")
opt = SGD(lr=0.01)
model = build_lenet(width=28, height=28, depth=1, classes=10,
                    weightsPath=args["weights"])

model.compile(loss="categorical_crossentropy",
              optimizer=opt, metrics=["accuracy"])

# visualize the model and save as model.png
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)


