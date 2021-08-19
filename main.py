from load_and_train import train_model
from gradcam import grad_cam
from util import *
from similarity_prop import find_similar_images
from add_images import retrain_model


for j in (4,5):
    train_model(0.5, j, num_epochs=10, toStoreQuery=True)
    print("grad_cam started")
    grad_cam(j, model_name=f'fmnist_run{j}') #model_name=f'model/mnist_{get_date()}'
    print("find_similar_images started")
    find_similar_images(j)
    print("retrain_model started")
    k_vals, accuracy = retrain_model(j, model_name=f'fmnist_run{j}')