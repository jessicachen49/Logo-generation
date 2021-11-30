from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.applications.vgg16 import preprocess_input
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from collections import defaultdict
import pickle
import cv2
import shutil
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
import numpy as np

def extract_vector(path):
    resnet_feature_list = []
    clean_imgs = []
    emp_count = []
    for file in os.listdir(path):
        im = cv2.imread(path+file)
        if im is not None and file not in clean_imgs:
            clean_imgs.append(file)
            im = cv2.resize(im,(224,224))
            img = np.expand_dims(im.copy(), axis=0)
            img = preprocess_input(img)
            resnet_feature = my_new_model.predict(img)
            resnet_feature_np = np.array(resnet_feature)
            resnet_feature_list.append(resnet_feature_np)
        else:
            emp_count.append(file) #list of images = None
    return resnet_feature_list, clean_imgs, emp_count

def visualize_clust(clustered_names, datadir):
    num = 30 #num of data to visualize from the cluster
    for j in range(len(clustered_names)):
        print(j)
        plt.figure(figsize=(15,15))
        for i in range(1,num): 
            plt.subplot(10, 10, i); #(Number of rows, Number of column per row, item number)
            img = datadir+clustered_names[j][i]
            if os.path.isfile(img):
                img = cv2.imread(img)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

datadir = '../data/clearbitwrds_wrds/' #directory of logo images
# extracting image features using VGG16
vgg_model = VGG16(include_top=True, weights='imagenet')
my_new_model = Model(inputs = vgg_model.input, outputs= vgg_model.layers[-2].output)
resnet_feat, clean_imgs, emp_count = extract_vector(datadir)
resnet_feat = np.vstack(np.array(resnet_feat))
# pca reduction
pca = PCA(n_components=1024)
resnet_pca = pca.fit_transform(resnet_feat)
# kmeans clustering
clust=7
x = resnet_pca
kmeans = KMeans(n_clusters = clust, random_state=0).fit(x)
y_kmeans = kmeans.predict(x)
# dictionary clustered_names[class] = list-of-image-names-in-class
clean_imgs = []
for i in os.listdir(datadir):
    clean_imgs.append(i)
clustered_names = {}
for i in range(clust):
    clustered_names[i] = []
# visualize clustered images
visualize_clust(clustered_names, datadir)

# creating pickle file for training
mypickle = defaultdict()
filepaths, labels = [], []
for i in range(clust):
    for file in os.listdir(datadir+str(i)):
        filepath = str(i) + '/' +file
        filepaths.append(filepath)
        labels.append(i)
        
mypickle['Filenames'] = filepaths
mypickle['Labels'] = labels
pickle.dump(mypickle, open("./data/mypickle.pickle", "wb")) 