import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

from keras.models import Model
from keras.models import load_model
from keras import backend as K

from util import get_date
import shutil
from pathlib import Path

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def grad_cam(run_no=3, model_name='mnist_2021-07-30', threshold = 0.5):
    #load model
    #model = load_model(f'model/mnist_{get_date()}')
    model = load_model('model/' + model_name)
    

    # Extract the outputs of the two conv layers
    layer_outputs = [layer.output for layer in model.layers[1:3]]
    # Create a model returning the layer_outputs for the model input
    act_model = Model(inputs=model.input, outputs=layer_outputs)


    #obtain query_images_y as array
    with open("query-images-y/run" + str(run_no) + "/labels.txt", "r") as f:
        #query_images_y = np.loadtxt(f.readlines())
        query_images_y = f.read()[1:-1].split(',')
    #query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
    query_images_y = [int(i) for i in query_images_y]


    #obtain query_images_index as array
    with open("query-images-y/run" + str(run_no) + "/indices.txt", "r") as f:
        query_images_index = f.read()[1:-1].split(',')
    query_images_index = [int(i) for i in query_images_index]

    X_test = np.load('dataset/run' + str(run_no) + '/X_test.npy')

    for i in range(len(query_images_y)):
        test_input = X_test[query_images_index[i]].reshape(1,X_test.shape[1],X_test.shape[2],1)

        which_number = query_images_y[i]
        # Get output vector
        output_vector = model.output[:, which_number]
        # Get the last convolutional layer
        last_conv_layer = model.layers[2]
        # Get the gradient of the given number with regard to the output feature map of the last conv layer
        grads = K.gradients(output_vector, last_conv_layer.output)[0] # (None,24,24,64)
        # Get the mean intensity of the gradient over each feature map (64)
        pooled_grads = K.mean(grads, axis=(0, 1, 2)) # 64
        # Compute gradient given an inputw
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([test_input]) # 64, (24,24,64)
        # Multiply each channel in the feature map array
        # by 'how important this channel is' with regard to the given number
        for j in range(len(pooled_grads_value)):
            conv_layer_output_value[:,:,j] *= pooled_grads_value[j]
        # Calculate channel-wise mean for the heatmap activation
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        
        heatmap = np.maximum(heatmap, 0) # => max(heatmap, 0)
        heatmap /= np.max(heatmap)

        #save heatmap and red channel of heatmap
        heatmap_saved = cv2.resize(heatmap, (28,28))
        heatmap_saved = np.uint8(255*heatmap_saved)
        heatmap_saved = cv2.applyColorMap(heatmap_saved, cv2.COLORMAP_JET)
        cv2.imwrite('query-images-grad-cam-heatmap/run' + str(run_no) + '/' + str(i) + '.jpg', heatmap_saved)
        red_heatmap = heatmap_saved.copy()
        red_heatmap[:, :, 0] = 0
        red_heatmap[:, :, 1] = 0
        
        cv2.imwrite('query-images-grad-cam-red-heatmap/run' + str(run_no) + '/' + str(i) + '.jpg', red_heatmap)
        red_heatmap = np.uint8(red_heatmap/255)
        
        
        #create input to compare to red heatmap
        input_compare = test_input.reshape(28,28,1)
        #input_compare = cv2.cvtColor(input_compare, cv2.COLOR_GRAY2BGR)
        #input_compare *= 255
        nonblack_count = 0 #from input image
        total_count = 0 #from heatmap how many red pixels
        for j in range(red_heatmap[:,:,2].shape[0]):
            for k in range(red_heatmap[:,:,2].shape[1]):
                if red_heatmap[j,k,2] == 1:
                    total_count += 1
                    if input_compare[j,k] > threshold:
                        nonblack_count += 1
        if i==49:
            print(red_heatmap)
            print(total_count)
        print(f"Query index: {i}")
        print(f"Count: {nonblack_count}")

grad_cam()