from load_and_train import train_model
from gradcam import grad_cam
from util import *
from similarity_prop import find_similar_images
from add_images import retrain_model

train_model(0.5, num_epochs=1, toStoreQuery=True)
grad_cam() #model_name=f'model/mnist_{get_date()}'
find_similar_images()
k_vals, accuracy = retrain_model()