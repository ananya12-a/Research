from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as k
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import os
import random

#load mnist dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data() #everytime loading data won't be so easy :)


#visualising first 9 data from training dataset
"""fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(X_train[i], cmap='gray', interpolation='none')
  plt.title("Digit: {}".format(y_train[i]))
  plt.xticks([])
  plt.yticks([])
fig"""

# let's print the actual data shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

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
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

print(np.unique(y_train, return_counts=True))

#set number of categories
num_category = 10

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_category)
y_test = keras.utils.to_categorical(y_test, num_category)
y_train[0]

X_train, X_add, y_train, y_add = train_test_split(X_train, y_train, train_size = 0.6)

def percentage_inc(original, new):
    return (new-original)/original * 100

def full_model(X_train, y_train, X_test, y_test, json_fname,  iter_nam, batch_size = 128, num_epoch = 10, toPlot = True):
    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)
    ##model building
    model = Sequential()
    #convolutional layer with rectified linear unit activation
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation='relu',
                    input_shape=input_shape))
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
    model.add(Dense(num_category, activation='softmax'))
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

    #how well did it do? 
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

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

    return score[1]

num_epoch = 10
accuracy_initial = full_model(X_train, y_train, X_test, y_test, "model_digit_before", "initial", num_epoch=num_epoch, toPlot=False)

k = int(0.5 * X_add.shape[0]) #number of samples to add
n = random.sample(range(0, X_add.shape[0]), k)
X_train_rand = np.append(X_train, np.zeros((k,X_train.shape[1],X_train.shape[2],X_train.shape[3])), axis=0)
y_train_rand = np.append(y_train, np.zeros((k,y_train.shape[1])), axis=0)
for i in range(k):
    X_train_rand[-1*i] = X_add[n[i]]
    y_train_rand[-1*i] = y_add[n[i]]
#print(X_train_rand[-1])
accuracy_rand = full_model(X_train_rand, y_train_rand, X_test, y_test, "model_digit_random", "random", num_epoch=num_epoch, toPlot=False)


print("percentage increase in  accuracy: " + str(percentage_inc(accuracy_initial, accuracy_rand)))