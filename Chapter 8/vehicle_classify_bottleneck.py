import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
from keras.models import load_model
import cv2
from keras.optimizers import SGD

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

    # model = applications.InceptionV3(include_top=False, weights='imagenet')

    datagen_train = ImageDataGenerator(
                            rescale=1. / 255,
                            rotation_range=40,
                            shear_range=0.2,
                            zoom_range=0.2,
                            horizontal_flip=True,
                            fill_mode='nearest')

    datagen = ImageDataGenerator(rescale=1. / 255)

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
    num_classes = len(generator.class_indices)

    predict_size_train = int(math.ceil(nb_train_samples / batch_size))

    bottleneck_features_train = model.predict_generator(
        generator, predict_size_train)

    np.save('data/models/bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    nb_validation_samples = len(generator.filenames)

    predict_size_validation = int(math.ceil(nb_validation_samples / batch_size))

    bottleneck_features_validation = model.predict_generator(
        generator, predict_size_validation)

    np.save('data/models/bottleneck_features_validation.npy', bottleneck_features_validation)


def train_top_model():
    datagen_top = ImageDataGenerator(rescale=1./255)
    generator_top = datagen_top.flow_from_directory(
            train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False)
    # print(generator_top.filenames)
    # print(generator_top.classes)

    # print(len(generator_top.filenames))
    # print(len(generator_top.classes))

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    np.save('data/models/class_indices.npy', generator_top.class_indices)

    # load the bottleneck features saved earlier
    train_data = np.load('data/models/bottleneck_features_train.npy')

    # get the class lebels for the training data, in the original order
    train_labels = generator_top.classes

    # https://github.com/fchollet/keras/issues/3467
    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    # print(train_labels)
    # print(train_data.shape)

    generator_top = datagen_top.flow_from_directory(
            validation_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode=None,
            shuffle=False)

    nb_validation_samples = len(generator_top.filenames)

    validation_data = np.load('data/models/bottleneck_features_validation.npy')

    validation_labels = generator_top.classes
    validation_labels = to_categorical(validation_labels, num_classes=num_classes)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # opt = SGD(lr=0.0005)
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=opt,
    #               loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='adam',
    #               loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)

    ######## Save the entire model to file ########
    #  model.save('my_model.h5')

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

    # image_path = 'data/validation/Cotton Pygmy Goose/Cotton Pygmy Goose (2).jpg'
    # image_path = 'data/eval/Cotton_Pygmy_Goose.jpg'
    # image_path = 'data/eval/Great_Cormorant.jpg'
    # image_path = 'data/eval/Lesser_Whistling_Duck.jpg'
    image_path = 'data/eval/00000596.jpg'

    orig = cv2.imread(image_path)

    print("[INFO] loading and preprocessing image...")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)

    # important! otherwise the predictions will be '0'
    image = image / 255

    # print(image.shape)

    image = np.expand_dims(image, axis=0)

    # print(image.shape)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    # get the bottleneck prediction from the pre-trained VGG16 model
    bottleneck_prediction = model.predict(image)

    # print(bottleneck_prediction)

    # build top model
    model = Sequential()
    model.add(Flatten(input_shape=bottleneck_prediction.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.load_weights(top_model_weights_path)

    ######## If the entire model is loaded from file ########
    # model = load_model('my_model.h5')

    # use the bottleneck prediction on the top model to get the final classification
    class_predicted = model.predict_classes(bottleneck_prediction)

    probabilities = model.predict_proba(bottleneck_prediction)

    # print(probabilities.argmax(axis=1))
    # print(probabilities[0, probabilities.argmax(axis=1)])

    # print(probabilities)

    prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]

    inID = class_predicted[0]

    # print(class_dictionary)
    inv_map = {v: k for k, v in class_dictionary.items()}
    # print(inv_map)

    label = inv_map[inID]

    # display the predictions to our screen
    print("Image ID: {}, Label: {}, Confidence: {}".format(inID, label, prediction_probability))

    plot_prediction_probabilities(probabilities, inv_map)

    # display the predictions to our screen
    cv2.putText(orig, "Predicted: {}".format(label), (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)
    # cv2.putText(orig, "Confidence: {:.5f}".format(prediction_probability), (10, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (43, 99, 255), 2)

    cv2.imshow("Classification", orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_prediction_probabilities(probabilities, label_map, top = 3):
    results = []
    for pred in probabilities:
        top_indices = pred.argsort()[-top:][::-1]
        result = [(label_map[i],) + (pred[i],) for i in top_indices]
        results.append(result)

    # print(results)

    plt.figure()
    order = list(reversed(range(len(results[0]))))
    bar_preds = [pr[1] for pr in results[0]]

    labels = (pr[0] for pr in results[0])
    # labels = (label_map[i] for i in list(range(len(results[0]))))

    plt.barh(order, bar_preds, alpha=0.5)
    plt.yticks(order, labels)
    plt.xlabel('Probability')
    # plt.xlim(0, 1.01)
    plt.tight_layout()
    plt.show()

save_bottlebeck_features()
train_top_model()
# predict()


cv2.destroyAllWindows()
