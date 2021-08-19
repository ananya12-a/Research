
import random
import keras
from keras.models import load_model
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def read_k(run_no):
    with open("AL_Results/run" + str(run_no) + "/k_vals.txt", "r") as f:
        k_vals = f.read()[1:-1].split(',')
    return k_vals

def add_random_images(run_no, model_name = 'mnist_2021-07-30', num_epochs=10):
    k_vals = read_k(run_no)
    accuracy = []
    if run_no == 1:
        run_no = 2
        X_add = np.load('dataset/run' + str(run_no) + '/X_add.npy')
        y_add = np.load('dataset/run' + str(run_no) + '/y_add.npy')
        X_train = np.load('dataset/run' + str(run_no) + '/X_train.npy')
        y_train = np.load('dataset/run' + str(run_no) + '/y_train.npy')
        X_test_final = np.load('dataset/run' + str(run_no) + '/X_test_final.npy')
        y_test_final = np.load('dataset/run' + str(run_no) + '/y_test_final.npy')
        y_add_max = np.array([np.argmax(rec) for rec in y_add])
        y_train_max =  np.array([np.argmax(rec) for rec in y_train])
        run_no=1
    else:
        X_add = np.load('dataset/run' + str(run_no) + '/X_add.npy')
        y_add = np.load('dataset/run' + str(run_no) + '/y_add.npy')
        X_train = np.load('dataset/run' + str(run_no) + '/X_train.npy')
        y_train = np.load('dataset/run' + str(run_no) + '/y_train.npy')
        X_test_final = np.load('dataset/run' + str(run_no) + '/X_test_final.npy')
        y_test_final = np.load('dataset/run' + str(run_no) + '/y_test_final.npy')
        y_add_max = np.array([np.argmax(rec) for rec in y_add])
        y_train_max =  np.array([np.argmax(rec) for rec in y_train])
    
    model = load_model('model/' + model_name)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
    accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1]) 
    for k in k_vals:
        index_of_image_to_add = random.sample(range(len(X_add)), int(k))
        X_train = np.insert(X_train, 0, X_add[index_of_image_to_add], axis = 0)
        y_train = np.insert(y_train, 0, y_add[index_of_image_to_add], axis = 0)
        X_add = np.delete(X_add, index_of_image_to_add, axis=0)
        y_add = np.delete(y_add, index_of_image_to_add, axis=0)
        y_add_max = np.array([np.argmax(rec) for rec in y_add])
        y_train_max =  np.array([np.argmax(rec) for rec in y_train])
        hist = model.fit(X_train, y_train,
                                batch_size=128,
                                epochs=num_epochs,
                                verbose=1)
        accuracy.append(model.evaluate(X_test_final, y_test_final, verbose=0)[1])
        print(accuracy[-1])
    with open('AL_Results/run' + str(run_no) + '/rand_accuracy.txt', 'w') as f:
        f.write(str(accuracy))


add_random_images(1, model_name='mnist_2021-07-30')
add_random_images(2, model_name='mnist_2021-08-06_run2')
add_random_images(3, model_name='fmnist_run3')
            