from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report,accuracy_score
from sklearn.decomposition import PCA
from skimage.feature import hog

import tensorflow as tf

import time
from datetime import timedelta
import os

import inception
from inception import transfer_values_cache

import cifar10
from cifar10 import num_classes

import os

def displayImageGrid(xtr, ytr):
    fig, axes1 = plt.subplots(10, 10, figsize=(8,8))

    for i, label in enumerate(class_names, start=0):
        itemsindex = np.where(ytr == i)
        randomImages = np.random.choice(itemsindex[0], 10);
        for j in range(10):
            index = randomImages[j]
            axes1[i][j].set_axis_off()
            axes1[i][j].imshow(xtr[index:index+1][0])    
        
    plt.show()

def hogAndSVM(xtr, ytr, xte, yte):
    hog_featurestr = []

    for image in xtr:
        fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, multichannel=True)
        hog_featurestr.append(fd)
        
    hog_featureste = []

    for image in xte:
        fd = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, multichannel=True)
        hog_featureste.append(fd)

    hog_featurestr = np.array(hog_featurestr)
    hog_featureste = np.array(hog_featureste)
    SVMPrint(hog_featurestr, ytr, hog_featureste, yte)

def SVMPrint(xTrain, yTrain, xTest, yTest):
    clf = SVC()
    clf.fit(xTrain, yTrain)
    
    yPred = clf.predict(xTest)

    print("Accuracy: "+ str(accuracy_score(yTest, yPred)))
    print ("\n")
    print (classification_report(yTest, yPred))

def plot_transfer_values(i):
    print('Input image: ')
    
    plt.imshow(images_test[i], interpolation='nearest')
    plt.show()
    
    print('Transfer-values for the image using Inception model:')
    
    img = transfer_values_test[i]
    img = img.reshape(32, 64)

    # Plot the image for the transfer-values.
    plt.imshow(img, interpolation='nearest', cmap='Reds')
    plt.show()
    
def plot_scatter(values, cls):
    import matplotlib.cm as cm
    cmap = cm.rainbow(np.linspace(0.0, 1.0, num_classes))
    
    colors = cmap[cls]
    
    x = values[:, 0]
    y = values[:, 1]
    
    plt.scatter(x, y, color=colors)
    plt.show()

#Download cifar images
cifar10.maybe_download_and_extract()


#assaign class names as a list
class_names = cifar10.load_class_names()
#assign images for training, class numbers , labels for training of size[numof images, num of class]
images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

#use only 10% of the trainnig image set, no gpu
#for hog, results increase from 32% to 45% when dataset increases from 1/10th to 1/5th
ratio = 10
images_train = images_train[0:int(len(images_train)/ratio)]
cls_train = cls_train[0:int(len(cls_train)/ratio)]
labels_train = labels_train[0:int(len(labels_train)/ratio)]
images_test =  images_test[0:int(len(images_test)/ratio)]
cls_test = cls_test[0:int(len(cls_test)/ratio)]
labels_test = labels_test[0:int(len(labels_test)/ratio)]

#displayImageGrid(images_train, cls_train)
#hogAndSVM(images_train, cls_train, images_test, cls_test)

#download inception model
inception.maybe_download()

#create inception model for transfer learning, with the layers
model = inception.Inception()

#path to save training after loading once
file_path_cache_train = os.path.join(cifar10.data_path, 'inception_cifar_train.pkl')
file_path_cache_test = os.path.join(cifar10.data_path, 'inception_cifar_test.pkl')

print ("Processing Inception transfer-values...")
images_scaled = images_train * 255.0
#transfer values are the CNN codes
transfer_values_train = transfer_values_cache(cache_path=file_path_cache_train, images=images_scaled, model=model)

print ("Processing Inception for test-images...")

images_scaled = images_test * 255.0
transfer_values_test = transfer_values_cache(cache_path=file_path_cache_test, images=images_scaled, model=model)
plot_transfer_values(i=16)

#embedd CNN codes in two dimensions
pca = PCA(n_components=2)
transfer_values_reduced = pca.fit_transform(transfer_values_train)

plot_scatter(transfer_values_reduced, cls_train)

#train a svm classifier to classify images using the transfer values and test on transfer_values_test

SVMPrint(transfer_values_train, cls_train, transfer_values_test, cls_test)



