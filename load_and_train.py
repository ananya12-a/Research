from __future__ import print_function
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
from pathlib import Path
import json
import shutil


def load_data(ratio, num_category=10):
    #load mnist dataset
    (X_train_temp, y_train_temp), (X_test_temp, y_test_temp) = fashion_mnist.load_data()
    X_train, X_add, y_train, y_add = train_test_split(X_train_temp, y_train_temp, train_size = ratio)
    X_test, X_test_final, y_test, y_test_final = train_test_split(X_test_temp, y_test_temp, train_size = ratio)    
    X_train, y_train, X_test, y_test, input_shape_train = data_formatting(X_train, y_train, X_test, y_test, num_category)
    X_add, y_add, X_test_final, y_test_final, input_shape_add = data_formatting(X_add, y_add, X_test_final, y_test_final, num_category)
    """
        A = X_train, y_train
        B = X_add, y_add
        C = X_test, y_test
        D = X_test_final, y_test_final
    """
    return X_train, X_add, y_train, y_add,X_test, X_test_final, y_test, y_test_final, input_shape_train

def data_formatting(X_train, y_train, X_test, y_test, num_category):
    #input image size 28*28
    img_rows , img_cols = 28, 28
    #reshaping
    #this assumes our data format
    #For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
    #"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).
    if k.image_data_format() == 'channels_first':
        X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
    #more reshaping
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2])
    y_train = keras.utils.to_categorical(y_train, num_category)
    y_test = keras.utils.to_categorical(y_test, num_category)

    return X_train, y_train, X_test, y_test, input_shape


# load and prepare the image
def load_image(img):
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	return img


def full_model(X_train, y_train, X_test, y_test, json_fname,  input_shape, iter_nam="", batch_size = 128, num_epoch = 10, toPlot = True, toStoreQuery = False, num_category = 10):
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)
    ##model building
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    #32 convolution filters used each of size 3x3
    #again
    model.add(Conv2D(64, (3, 3), activation='relu'))
    #64 convolution filters used each of size 3x3
    #choose the best features via pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #randomly turn neurons on and off to improve convergence
    model.add(Dropout(0.25))
    #flatten since too many dimensions, we only want a classification output
    model.add(Flatten())
    #fully connected to get all relevant data
    model.add(Dense(128, activation='relu'))
    #one more dropout for convergence' sake :) 
    model.add(Dropout(0.5))
    #output a softmax to squash the matrix into output probabilities
    model.add(Dense(num_category, activation='softmax', name='visual_layer'))
    #Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
    #categorical ce since we have multiple classes (10) 
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])


    #model training
    model_log = model.fit(X_train, y_train,
            batch_size=batch_size,
            epochs=num_epoch,
            verbose=1,
            validation_data=(X_test, y_test))
    
    #model accuracy calculation
    """correctly_classified = 0
    query_images_X = []
    query_images_y = []
    for i in range(X_test.shape[0]):
        img = load_image(X_test[i])
        if np.argmax(y_test[i]) == np.argmax(model.predict(img)):
            correctly_classified += 1
        elif toStoreQuery:
            query_images_X.append(X_test[i])
            query_images_y.append(y_test[i])
    accuracy = correctly_classified/X_test.shape[0]"""
    
    
    query_images_y=[]
    y_pred = model.predict(X_test)
    y_pred = np.array([np.argmax(rec) for rec in y_pred])
    y_test_max = np.array([np.argmax(rec) for rec in y_test])
    accuracy = sum(y_pred == y_test_max)/len(y_test)
    if toStoreQuery:
        res = [i for i, val in enumerate(y_pred != y_test_max) if val]
        count = 0
        shutil.rmtree('/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images')
        Path("/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images").mkdir(parents=True, exist_ok=True)
        Path("/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images-y").mkdir(parents=True, exist_ok=True)
        for i in res:
            img = load_image(X_test[i])
            print(img.shape)
            img_array = X_test[i] * 255
            cv2.imwrite('query-images/' + str(count) + '.jpg', img_array)
            query_images_y.append(y_test[i])
            count += 1
        with open('query-images-y/labels.txt', 'w') as f:
            for item in query_images_y:
                item = str(item)[1:-1]
                f.write("%s\n" % item)
        
            
    
    #model evaluation
    """
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    accuracy = score[1]"""

    if toPlot:
        # plotting the metrics
        fig = plt.figure()
        fig = plt.subplot(2,1,1)
        fig = plt.plot(range(num_epoch), model_log.history['accuracy'])
        fig = plt.plot(range(num_epoch), model_log.history['val_accuracy'])
        fig = plt.title('model accuracy ' + iter_nam)
        fig = plt.ylabel('accuracy')
        fig = plt.xlabel('epoch')
        fig = plt.legend(['train', 'test'], loc='lower right')

        fig = plt.subplot(2,1,2)
        fig = plt.plot(model_log.history['loss'])
        fig = plt.plot(model_log.history['val_loss'])
        fig = plt.title('model loss '+ iter_nam)
        fig = plt.ylabel('loss')
        fig = plt.xlabel('epoch')
        fig = plt.legend(['train', 'test'], loc='upper right')

        fig = plt.tight_layout()

        plt.show(block=True)

    #Save the model
    # serialize model to JSON
    model_digit_json = model.to_json()
    with open(json_fname + '.json', 'w') as json_file:
        json_file.write(model_digit_json)
    # serialize weights to HDF5
    model.save_weights(json_fname + '.h5')
    print('Saved model to disk')

    return accuracy, model


"""
        A = X_train, y_train
        B = X_add, y_add
        C = X_test, y_test
        D = X_test_final, y_test_final
    """
