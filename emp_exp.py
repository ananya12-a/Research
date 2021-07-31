
from keras.models import load_model
import shutil
from pathlib import Path
import numpy as np
import keras
import random
from glob import glob
import scipy.spatial
import cv2

from util import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def Nmaxelements_cond(list1, N, y_add, query_y_value):
    final_list = []
    index_list = []
  
    for i in range(0, N): 
        max1 = 0
        max_index = 0
          
        for j in range(len(list1)):     
            if list1[j] > max1 and y_add[j] == query_y_value:
                max1 = list1[j]
                max_index = j
        
        if max1 in list1:
            list1.remove(max1)
        final_list.append(max1)
        index_list.append(max_index)
          
    return index_list


with open("ratios/ratios.txt", "r") as f:
        #query_images_y = np.loadtxt(f.readlines())
        ratios = f.read()[1:-1].split(',')
#query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
ratios = [float(i) for i in ratios]



with open("query-images-y/labels.txt", "r") as f:
        #query_images_y = np.loadtxt(f.readlines())
        query_images_y = f.read()[1:-1].split(',')
    #query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
query_images_y = [int(i) for i in query_images_y]
query_images_fn = glob('query-images-final/*')

def quadratic(model_name='mnist_2021-07-30', num_epochs = 5, tol = 1E-2):
    X_test = np.load('dataset/X_test.npy')
    y_test = np.load('dataset/y_test.npy')
    X_add = np.load('dataset/X_add.npy')
    y_add = np.load('dataset/y_add.npy')
    X_train = np.load('dataset/X_train.npy')
    y_train = np.load('dataset/y_train.npy')
    shutil.rmtree('model/linear')
    Path("model/linear").mkdir(parents=True, exist_ok=True)
    #y = ax + b
    model = load_model('model/' + model_name)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    accuracy_int = model.evaluate(X_test, y_test, verbose=0)[1]
    a_old = random.randint(0,100)
    b_old = random.randint(0,100)
    c_old = random.randint(0,100)
    a = a_old + random.randint(0,5)
    b = b_old + random.randint(0,5)
    c = c_old + random.randint(0,5)
    models = []
    print(a,b,c)
    for j in range(10):
        accuracy = [accuracy_int]
        total_to_add = 0
        for i in range(len(ratios)):
            to_add = int(a*pow(ratios[i],2) + b*ratios[i] + c)
            total_to_add += to_add
            query_image = cv2.imread(query_images_fn[i])
            query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
            similarity = []
            query_image_num = int(query_images_fn[i][query_images_fn[i].index('/')+1:-4])
            query_image_y = query_images_y[query_image_num]
            for k in range(X_add.shape[0]):
                similarity.append(scipy.spatial.distance.cosine(query_image.flatten(), X_add[k].flatten()))
            y_add_max = np.array([np.argmax(rec) for rec in y_add])
            index_vals = Nmaxelements_cond(similarity, to_add, y_add_max, query_image_y)    
            X_train = np.insert(X_train, 0, X_add[index_vals], axis = 0)
            X_add = np.delete(X_add, index_vals, 0)
            y_train = np.insert(y_train, 0, y_add[index_vals], axis = 0)
            y_add = np.delete(y_add, index_vals, 0)
            """for k in index_vals:
                X_train = np.insert(X_train, 0, X_add[k], axis = 0)
                X_add = np.delete(X_add, k, 0)
                y_train = np.insert(y_train, 0, y_add[k], axis = 0)
                y_add = np.delete(y_add, k, 0)"""
                #X_train.append(X_add[k])
                #y_train.append(y_add[k])
            # Run model
        hist = model.fit(X_train, y_train,
                            batch_size=128,
                            epochs=num_epochs,
                            verbose=1)
            # Save model
            
        accuracy_curr = model.evaluate(X_test, y_test, verbose=0)[1]

        if accuracy_curr - accuracy[-1] < tol:
            a = a_old
            b = b_old
            c = c_old
            X_train = X_train[total_to_add:]
            y_train = y_train[total_to_add:]
            j -= 1
        else:
            a_old = a
            b_old = b
            c_old = c
            a = a_old + random.randint(0,5)
            b = b_old + random.randint(0,5)
            c = c_old + random.randint(0,5)
            accuracy.append(accuracy_curr)
            models.append(model)
        print(a,b,c)
    models[-1].save(f'model/linear/mnist_{get_date()}')

    print("FINAL")
    print(a,b,c)


quadratic()