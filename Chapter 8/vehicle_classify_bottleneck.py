import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
import cv2

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'data/models/bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

# number of epochs to train top model
epochs = 50
# batch size used by flow_from_directory and predict_generator
batch_size = 16


def save_bottlebeck_features():
    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    ### save the bottleneck features for the training data ###

    datagen_train = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    generator = datagen_train.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    print(len(generator.filenames))
    print(generator.class_indices)
    print(len(generator.class_indices))

    nb_train_samples = len(generator.filenames)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('data/models/bottleneck_features_train.npy',
            bottleneck_features_train)

    ### save the bottleneck features for the validation data ###

    datagen_validation = ImageDataGenerator(rescale=1. / 255)

    generator = datagen_validation.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(
        math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('data/models/bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    # use a data generator to load the labels for the training and validation data
    datagen_top = ImageDataGenerator(rescale=1./255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    num_classes = len(generator_top.class_indices)

    # save the class indices for use in the predictions
    np.save('data/models/class_indices.npy', generator_top.class_indices)

    # load the bottleneck features for the train data saved earlier
    train_data = np.load('data/models/bottleneck_features_train.npy')

    # get the class labels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    # load the bottleneck features for the validation data saved earlier
    validation_data = np.load('data/models/bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1)

    print("\n")

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    print("\n")
    print("\n")
    print(generator_top.class_indices)

    plt.figure(1)

    # summarize history for accuracy

    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def predict():
    class_dictionary = np.load('data/models/class_indices.npy').item()

    num_classes = len(class_dictionary)

    print("[INFO] loading and preprocessing image...")
    image_path = 'data/eval/00000520.jpg'

    orig = cv2.imread(image_path)
    image = load_img(image_path, target_size=(img_width, img_height))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255.0

    # add a new axis to make the image array confirm with
    # the (samples, height, width, depth) structure
    image = np.expand_dims(image, axis=0)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.load_weights(top_model_weights_path)

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    # get the probabilities for the prediction
    probabilities = model.predict_proba(bottleneck_prediction)

    prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]

    inID = class_predicted[0]

    # invert the class dictionary in order to get the label for the id
    inv_map = {v: k for k, v in class_dictionary.items()}
    label = inv_map[inID]

    # display the prediction in the console
    print("Image ID: {}, Label: {}, Confidence: {}".format(
        inID, label, prediction_probability))

    # display the prediction in OpenCV window
    cv2.putText(orig, "Predicted: {}".format(label), (10, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (43, 99, 255), 3, cv2.LINE_AA)

    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


### Uncomment as necessary
# save_bottlebeck_features()
# train_top_model()
predict()


cv2.destroyAllWindows()
