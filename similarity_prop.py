from glob import glob
import cv2
import numpy as np
import scipy.spatial
import image_similarity_measures
from image_similarity_measures.quality_metrics import rmse, ssim, sre

def Nmaxelements(list1, N):
    final_list = []
    index_list = []
  
    for i in range(0, N): 
        max1 = 0
        max_index = 0
          
        for j in range(len(list1)):     
            if list1[j] > max1:
                max1 = list1[j]
                max_index = j
                  
        list1.remove(max1)
        final_list.append(max1)
        index_list.append(max_index)
          
    return index_list

def find_similar_images():
    query_images_fn = glob('query-images-final/*')
    X_train = np.load('dataset/X_train.npy')
    y_train = np.load('dataset/y_train.npy')
    y_train =  np.array([np.argmax(rec) for rec in y_train])
    #obtain query_images_y as array
    with open("query-images-y/labels.txt", "r") as f:
        #query_images_y = np.loadtxt(f.readlines())
        query_images_y = f.read()[1:-1].split(',')
    #query_images_y = np.array([np.argmax(rec) for rec in query_images_y])
    query_images_y = [int(i) for i in query_images_y]
    ratio = []
    for i in range(len(query_images_fn)):
        query_image = cv2.imread(query_images_fn[i])
        query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
        similarity = []
        for j in range(X_train.shape[0]):
            #X_train_inst = cv2.cvtColor(X_train[j], cv2.COLOR_GRAY2BGR)
            #print(X_train_inst.shape)
            similarity.append(scipy.spatial.distance.cosine(query_image.flatten(), X_train[j].flatten()))
        index_vals = Nmaxelements(similarity, 50)
        class_count = 0
        query_image_num = int(query_images_fn[i][query_images_fn[i].index('/')+1:-4])
        query_image_y = query_images_y[query_image_num]
        for j in range(len(index_vals)):
            #cv2.imwrite('temp/' + str(j) + '.jpg', X_train[index_vals[j]]*255)
            if y_train[index_vals[j]] == query_image_y:
                class_count += 1
        ratio.append(class_count/len(index_vals))
        #break
        #cos_sim=np.dot(query_2D,B)/(np.linalg.norm(query_2D)*np.linalg.norm(B))
    with open('ratios/ratios.txt', 'w') as f:
        f.write(str(ratio))


find_similar_images()