import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DIR = 'dataset/PetImages'
IMG_SIZE = 100
categories = ['Dog', 'Cat']
def readData(directory):
    data = []
    for category in categories:
            path = os.path.join(directory, category)
            classification = categories.index(category)
            file_list = os.listdir(path)
            for img in file_list:
                try:
                    #img = str(file_list.index(img)) + '.jpg' 
                    check_path = os.path.exists(os.path.join(path, img))
                    img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    if img_array is None:
                        raise FileNotFoundError('image ' + img + ' not found in path ' + path)
                    #print(img_array.shape)
                    #plt.imshow(img_array, cmap='gray')
                    #plt.show()
                    resized_img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    #plt.imshow(resized_img_array, cmap='gray')
                    #plt.show()
                    data.append([resized_img_array, classification])
                except Exception as exception:
                    print('An exception occured with error: ' + str(exception))
                    continue
    return data
def featureLabelSplit(data):
    X = []
    y = []
    for features, labels in data:
         X.append(features)
         y.append(labels)
    X_array = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    return X_array, y
def preprocessData(directory):
    data = readData(directory)
    random.shuffle(data)
    features, labels = featureLabelSplit(data)
    with open('features.pickle', 'wb') as file:
        pickle.dump(features, file)
    with open('labels.pickle', 'wb') as file:
        pickle.dump(labels, file)
        

preprocessData(DIR)