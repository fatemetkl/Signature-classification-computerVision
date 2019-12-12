from feature import GetFeature
from NeuralNetwork import neuralnet
import os
import cv2
from os import walk
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend


def save_data_for_each_input():
    here = os.getcwd()
    features = []
    data = []
    labels = []
    images = []
    ignore=0
    for i in range(0, 10):
        path = os.path.join(here, "images\\dataset", str(i))
        for (dirpath, dirnames, filenames) in walk(path):
            for file in filenames:
                cur_path = os.path.join(path, file)
                gray = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2GRAY)
                images.append(gray)
                labels.append(i)


    feature = GetFeature(20, 10)
    final_labels=[]
    i=-1
    for img in images:
        i += 1
        try:
            data.append(feature.describe(img))
            final_labels.append(labels[i])
        except:
            continue
    with open('data.txt', 'w') as filehandle:
        for listitem in data:
            filehandle.write('%s\n' % listitem)

    with open("label.txt","w") as labels:
        for item in final_labels:
            labels.write('%s\n' % item)
    return ("done")


def fire_neuralnetwork(feature_np,label_np):
    ohe = OneHotEncoder()
    y = ohe.fit_transform(label_np).toarray()
    X_train, X_test, y_train, y_test = train_test_split(feature_np, y, test_size=0.1)  # 10%train
    NeuralNet = neuralnet(3)
    # train
    model = NeuralNet.train(X_train, y_train)

    NeuralNet.test(model, X_test, y_test)



#execute this one time :))) for the sake of time
#This function saves the input features and labels into data.txt and label.txr




# print(save_data_for_each_input())

# here we read that files
features=[]
with open('data.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1].replace("[","").replace("]","")
        features.append(currentPlace)


lables=[]
with open('label.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        lables.append(currentPlace)
final_feature=[]
for item in features:
    if len(item.split(' '))>2:
        item=item.split(" ")
        if(len(item)==5):
            item=item[1:5]
        for x in item:
            final_feature.append(x)

    else:
        final_feature.append(item)



lables=[float(i) for i in lables]
features=[float(i[:6]) for i in final_feature]

feature_np = np.asarray(features, dtype=np.float32)
label_np = np.asarray(lables, dtype=np.float32)
label_np=label_np.reshape(-1,1)# important
feature_np=feature_np.reshape(-1,25)

# fire_neuralnetwork(feature_np,label_np)

