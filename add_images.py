from glob import glob
import cv2
import numpy as np
import scipy.spatial
import keras
from keras.models import load_model
from util import *
from similarity_prop import Nmaxelements

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def Nmaxelements_cond(list1, N, y_add, query_y_value):
    final_list = []
    index_list = []
  
    for i in range(0, N): 
        max1 = -1
        max_index = -1
          
        for j in range(len(list1)):     
            if list1[j] > max1 and y_add[j] == query_y_value:
                max1 = list1[j]
                max_index = j
        
        if max1 in list1:
            list1.remove(max1)
        final_list.append(max1)
        index_list.append(max_index)
          
    return index_list

def retrain_model(run_no, model_name = 'mnist_2021-07-30', a = 28, b = 37, c = 32, num_epochs = 10, ratio_threshold = 0.35, score_threshold = 500):
    k_vals = []
    accuracy = [] #update after every query image
    X_add = np.load('dataset/run' + str(run_no) + '/X_add.npy')
    y_add = np.load('dataset/run' + str(run_no) + '/y_add.npy')

    X_train = np.load('dataset/run' + str(run_no) + '/X_train.npy')
    y_train = np.load('dataset/run' + str(run_no) + '/y_train.npy')
    X_test_final = np.load('dataset/run' + str(run_no) + '/X_test_final.npy')
    y_test_final = np.load('dataset/run' + str(run_no) + '/y_test_final.npy')

    """with open("ratios/run" + str(run_no) + "/scores.txt", "r") as f:
            #query_images_y = np.loadtxt(f.readlines())
            scores = f.read()[1:-1].split(',')
    #query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
    scores = [int(i) for i in scores]"""


    with open("ratios/run" + str(run_no) + "/ratios.txt", "r") as f:
            #query_images_y = np.loadtxt(f.readlines())
            ratios = f.read()[1:-1].split(',')
    #query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
    ratios = [float(i) for i in ratios]

    with open("query-images-y/run" + str(run_no) + "/labels.txt", "r") as f:
            #query_images_y = np.loadtxt(f.readlines())
            query_images_y = f.read()[1:-1].split(',')
        #query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
    query_images_y = [int(i) for i in query_images_y]
    query_images_fn = glob('query-images-final/run' + str(run_no) + '/*')


    model = load_model('model/' + model_name)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1])
    print(accuracy)
    """y_add_max = np.array([np.argmax(rec) for rec in y_add])
    y_train_max =  np.array([np.argmax(rec) for rec in y_train])
    for i in range(len(ratios)):
        score = scores[i]
        scores_int = [score]
        no_added = 0
        query_image = cv2.imread(query_images_fn[i])
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        query_image_y = query_images_y[int(query_images_fn[i][query_images_fn[i].index('/')+1:-4])]
        print(query_image_y)
        while score < score_threshold:
            no_added += 1
            similarity = []

            #calculate similarity between X_add images and query image
            for j in range(X_add.shape[0]):
                similarity.append(scipy.spatial.distance.cosine(query_image.flatten(), X_add[j].flatten()))
            #find the most similar image in X_add with the same class
            index_of_image_to_add = Nmaxelements_cond(similarity, 1, y_add_max, query_image_y)[0] #Nmaxelements(similarity, 1)[0]

            #move the most similar image from X_add to X_train
            X_train = np.insert(X_train, 0, X_add[index_of_image_to_add], axis = 0)
            y_train = np.insert(y_train, 0, y_add[index_of_image_to_add], axis = 0)
            X_add = np.delete(X_add, index_of_image_to_add, axis=0)
            y_add = np.delete(y_add, index_of_image_to_add, axis=0)
            print(X_train.shape)
            print(X_add.shape)



            similarity = []
            #calculate similarity between X_train images and query image
            for j in range(X_train.shape[0]):
                similarity.append(scipy.spatial.distance.cosine(query_image.flatten(), X_train[j].flatten()))
            
            #get 50 most similar X_train images
            index_vals = Nmaxelements(similarity, 50)

            #recalculate ratio
            class_count = 0
            for j in range(len(index_vals)):
                if y_train_max[index_vals[j]] == query_image_y:
                    class_count += len(index_vals)-j
            score = class_count #/len(index_vals)
            scores_int.append(score)
            print("score: %f"%score)
            '''if scores[-1] == scores[-2] and scores[-1] == scores[-1]:
                break'''
        k_vals.append(no_added)
        hist = model.fit(X_train, y_train,
                                batch_size=128,
                                epochs=num_epochs,
                                verbose=1)
        accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1])
        print(accuracy[-1])"""
    y_add_max = np.array([np.argmax(rec) for rec in y_add])
    y_train_max =  np.array([np.argmax(rec) for rec in y_train])
    X_train_og = X_train
    y_train_og = y_train
    for i in range(len(ratios)):
        scores = [ratios[i]]
        no_added = 0
        query_image = cv2.imread(query_images_fn[i])
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        query_image_y = query_images_y[int(query_images_fn[i][query_images_fn[i].index('/')+6:-4])]
        print(query_image_y)
        while scores[-1] < ratio_threshold:
            no_added += 1
            similarity = []

            #calculate similarity between X_add images and query image
            for j in range(X_add.shape[0]):
                similarity.append(scipy.spatial.distance.cosine(query_image.flatten(), X_add[j].flatten()))
            #find the most similar image in X_add with the same class
            index_of_image_to_add = Nmaxelements_cond(similarity, 1, y_add_max, query_image_y)[0] #Nmaxelements(similarity, 1)[0]

            #move the most similar image from X_add to X_train
            X_train = np.insert(X_train, 0, X_add[index_of_image_to_add], axis = 0)
            y_train = np.insert(y_train, 0, y_add[index_of_image_to_add], axis = 0)
            X_add = np.delete(X_add, index_of_image_to_add, axis=0)
            y_add = np.delete(y_add, index_of_image_to_add, axis=0)
            y_add_max = np.array([np.argmax(rec) for rec in y_add])
            y_train_max =  np.array([np.argmax(rec) for rec in y_train])
            print(X_train.shape)
            print(X_add.shape)



            similarity = []
            #calculate similarity between X_train images and query image
            for j in range(X_train.shape[0]):
                similarity.append(scipy.spatial.distance.cosine(query_image.flatten(), X_train_og[j].flatten()))
            
            #get 50 most similar X_train images
            index_vals = Nmaxelements(similarity, 50)

            #recalculate ratio
            class_count = 0
            for index in index_vals:
                if y_train_max[index] == query_image_y:
                    class_count += 1
            scores.append(class_count/len(index_vals))
            print("score: %f"%scores[-1])
            print(scores[-6:-1])
            print(scores[-5:])
            if scores[-6:-1] == scores[-5:]:
                break
        k_vals.append(no_added)
        hist = model.fit(X_train, y_train,
                                batch_size=128,
                                epochs=num_epochs,
                                verbose=1)
        accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1])
        print(accuracy[-1])
    """for i in range(len(ratios)):
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
        '''if X_train.shape[0] + X_add.shape[0] != 60000 or y_train.shape[0] + y_add.shape[0] != 60000:
            raise Exception(str(X_train.shape[0] + X_add.shape[0]) + " total samples instead of 60,000")'''
        hist = model.fit(X_train, y_train,
                                batch_size=128,
                                epochs=num_epochs,
                                verbose=1)
        accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1])
        print(accuracy[-1])
    model.save(f'model/final/fmnist_{get_date()}')
    print('Model saved!')"""
    with open('AL_Results/run' + str(run_no) + '/k_vals.txt', 'w') as f:
        f.write(str(k_vals))
    with open('AL_Results/run' + str(run_no) + '/accuracy.txt', 'w') as f:
        f.write(str(accuracy))
    return k_vals, accuracy
"""k_vals, accuracy = retrain_model()
print("Results")
print(k_vals)
print(accuracy)"""