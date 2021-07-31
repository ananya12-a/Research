from glob import glob
import cv2
import numpy as np
import scipy.spatial
import keras
from keras.models import load_model
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

def retrain_model(model_name = 'mnist_2021-07-30', a = 28, b = 37, c = 32, num_epochs = 10):
    k_vals = []
    accuracy = [] #after every query image
    X_add = np.load('dataset/X_add.npy')
    y_add = np.load('dataset/y_add.npy')
    X_train = np.load('dataset/X_train.npy')
    y_train = np.load('dataset/y_train.npy')
    X_test_final = np.load('dataset/X_test_final.npy')
    y_test_final = np.load('dataset/y_test_final.npy')

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

    model = load_model('model/' + model_name)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1])
    for i in range(len(ratios)):
        to_add = int(a*pow(ratios[i],2) + b*ratios[i] + c)
        print(to_add)
        k_vals.append(to_add)
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
        y_train = np.insert(y_train, 0, y_add[index_vals], axis = 0)
        for j in index_vals:
            X_add = np.delete(X_add, j, axis=0)
            y_add = np.delete(y_add, j, axis=0)
        """if X_train.shape[0] + X_add.shape[0] != 60000 or y_train.shape[0] + y_add.shape[0] != 60000:
            raise Exception(str(X_train.shape[0] + X_add.shape[0]) + " total samples instead of 60,000")"""
        hist = model.fit(X_train, y_train,
                                batch_size=128,
                                epochs=num_epochs,
                                verbose=1)
        accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1])
        print(accuracy[-1])
    model.save(f'model/final/fmnist_{get_date()}')
    print('Model saved!')
    with open('AL_Results/k_vals.txt', 'w') as f:
        f.write(str(k_vals))
    with open('AL_Results/accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    return k_vals, accuracy
k_vals, accuracy = retrain_model()
print("Results")
print(k_vals)
print(accuracy)