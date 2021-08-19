#adapted from: https://github.com/jaekookang/mnist-grad-cam


import keras
from keras.datasets import fashion_mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Input
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from util import get_date, load_image

import numpy as np
import shutil
from pathlib import Path

import cv2

def prepare_fmnist(ratio, run_no, num_classes=10):
    # Prepare fmnist dataset
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    nrow, ncol = 28, 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, nrow, ncol)
        x_test = x_test.reshape(x_test.shape[0], 1, nrow, ncol)
        input_shape = (1, nrow, ncol)
    else:
        x_train = x_train.reshape(x_train.shape[0], nrow, ncol, 1)
        x_test = x_test.reshape(x_test.shape[0], nrow, ncol, 1)
        input_shape = (nrow, ncol, 1)
    # Change variable type, value range
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    # Convert classes into one-hot vector
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    X_train, X_add, y_train, y_add = train_test_split(
        x_train, y_train, train_size=ratio)
    X_test, X_test_final, y_test, y_test_final = train_test_split(
        x_test, y_test, train_size=ratio)
    np.save('dataset/run' + str(run_no) + '/X_train', X_train)
    np.save('dataset/run' + str(run_no) + '/X_test', X_test)
    np.save('dataset/run' + str(run_no) + '/X_add', X_add)
    np.save('dataset/run' + str(run_no) + '/X_test_final', X_test_final)
    np.save('dataset/run' + str(run_no) + '/y_train', y_train)
    np.save('dataset/run' + str(run_no) + '/y_test', y_test)
    np.save('dataset/run' + str(run_no) + '/y_add', y_add)
    np.save('dataset/run' + str(run_no) + '/y_test_final', y_test_final)
    return (X_train, y_train), (X_test, y_test), (X_add, y_add), (X_test_final, y_test_final), input_shape

def build_cnn(input_shape, num_classes):
    x = Input(shape=input_shape)
    h = Conv2D(filters=6, kernel_size=(3, 3), activation='tanh', input_shape=(28,28,1))(x)
    h = AveragePooling2D()(h)
    h = Conv2D(filters=16, kernel_size=(3, 3), activation='tanh')(h)
    h = AveragePooling2D()(h)
    h = Flatten()(h)
    h = Dense(units=128, activation='tanh')(h)
    y = Dense(num_classes, activation='softmax')(h)
    model = Model(inputs=[x], outputs=[y])
    return model

def train_model(ratio, run_no, num_epochs = 25, toStoreQuery=False):
    # Get mnist dataset
    (X_train, y_train), (X_test, y_test), (X_add, y_add), (X_test_final, y_test_final), input_shape = prepare_fmnist(ratio, run_no)
    # Build & compile model
    model = build_cnn(input_shape, 10)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    # Run model
    hist = model.fit(X_train, y_train,
                     batch_size=128,
                     epochs=num_epochs,
                     verbose=1)
    # Save model
    model.save(f'model/fmnist_run{run_no}')
    print('Model saved!')

    query_images_y = []
    query_images_index = []
    y_pred = model.predict(X_test)
    y_pred = np.array([np.argmax(rec) for rec in y_pred])
    y_test_max = np.array([np.argmax(rec) for rec in y_test])
    accuracy = sum(y_pred == y_test_max)/len(y_test_max)
    if toStoreQuery:
        res = [i for i, val in enumerate(y_pred != y_test_max) if val]
        count = 0
        shutil.rmtree('/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images/run' + str(run_no))
        Path("/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images/run" + str(run_no)).mkdir(parents=True, exist_ok=True)
        Path("/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images-y/run" + str(run_no)).mkdir(parents=True, exist_ok=True)
        for i in res:
            img = load_image(X_test[i])
            img_array = X_test[i] * 255
            cv2.imwrite('query-images/run' + str(run_no) + '/' + str(count) + '.jpg', img_array)
            query_images_y.append(y_test_max[i])
            query_images_index.append(i)
            count += 1
        with open('query-images-y/run' + str(run_no) + '/labels.txt', 'w') as f:
            f.write(str(query_images_y))
            """for item in query_images_y:
                item = str(item)[1:-1]
                f.write("%s\n" % item)"""
        with open('query-images-y/run' + str(run_no) + '/indices.txt', 'w') as f:
            f.write(str(query_images_index))
            """for item in query_images_index:
                item = str(item)[1:-1]
                f.write("%s\n" % item)"""

#train_model(0.5, num_epochs=10, toStoreQuery=True)