import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.initializers import glorot_uniform
from keras.datasets import fashion_mnist
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from pathlib import Path

import cv2
from pytorch_grad_cam.utils.image import show_cam_on_image


from tensorflow.keras.applications import imagenet_utils
import numpy as np
import argparse
import imutils
import cv2



def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()


    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")


    def compute_heatmap(self, image, eps=1e-8, img_size):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        guidedGrads = tf.reshape(guidedGrads,(-1, 1))


        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        cam = np.array(cam)
        heatmap = cv2.resize(cam, img_size)
        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        # return the resulting heatmap to the calling function
        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5,
                        colormap=cv2.COLORMAP_VIRIDIS):
        # apply the supplied color map to the heatmap and then
        # overlay the heatmap on the input image
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
        # return a 2-tuple of the color mapped heatmap and the output,
        # overlaid image
        return (heatmap, output)


def gradCAM_func(last_conv_layer_name, img_size):

    ##LOAD MODEL FROM WEIGHTS
    #Reading the model from JSON file
    with open('model_digit_before.json', 'r') as json_file:
        json_savedModel= json_file.read()
    #load the model architecture 
    model = tf.keras.models.model_from_json(json_savedModel)
    model.load_weights('model_digit_before.h5')
    #Compiling the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])


    ##PREPROCESSING FOR GRADCAM
    preprocess_input = keras.applications.xception.preprocess_input
    decode_predictions = keras.applications.xception.decode_predictions

    
    #obtain query_images_y as array
    with open("query-images-y/labels.txt", "r+") as f:
        query_images_y = np.loadtxt(f.readlines())
    query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
    
    Path("/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images-grad-cam").mkdir(parents=True, exist_ok=True)
    for i in range(len(query_images_y)):
        img_path = '/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images/' + str(i) + '.jpg'

        img_cnn = cv2.imread(img_path)
        img_cnn = cv2.cvtColor(img_cnn, cv2.COLOR_BGR2GRAY)

        img_cnn = img_cnn/ 255
        img_cnn = cv2.resize(img_cnn, img_size)
        img_cnn = img_cnn.reshape(1,img_size[0],img_size[1],1)

        preds = model.predict(img_cnn) 
        index = np.argmax(preds[0])
        


        icam = GradCAM(model, index, last_conv_layer_name) 
        heatmap = icam.compute_heatmap(img_cnn, img_size)
        heatmap = cv2.resize(heatmap, img_size)


        (heatmap, output) = icam.overlay_heatmap(heatmap, cv2.imread(img_path), alpha=0.5)

        cv2.imwrite('query-images-grad-cam/' + str(i) + '.jpg', output)


        #img_array_gray = img_array_gray.reshape(1,28,28,1)
        #preds = model.predict(img_array_gray)
        #heatmap = make_gradcam_heatmap(img_array_gray, model, last_conv_layer_name)
        #save_and_display_gradcam('/Users/ananyaaggarwal/Desktop/grad-cam-tf/query-images-grad-cam/' + str(i) + '.jpg', heatmap, alpha = 0.9)

img_size = (28,28)
last_conv_layer_name = "visual_layer"
gradCAM_func(last_conv_layer_name, img_size)