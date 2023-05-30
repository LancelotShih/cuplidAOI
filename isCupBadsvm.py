import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# dir = r'C:\Users\lance\OneDrive\Documents\College yr2\OView Code\cupInspectAOI\processedCup'

# categories = ['badProc', 'goodProc']

# data = []

# for category in categories:
#     path = os.path.join(dir, category)
#     label = categories.index(category)

#     for img in os.listdir(path):
#         imgpath = os.path.join(path,img)
#         goodbad_img = cv2.imread(imgpath,0)
#         try:
#             goodbad_img = cv2.resize(goodbad_img,(50,50))
#             image = np.array(goodbad_img).flatten()

#             data.append([image,label])
#         except Exception as e:
#             pass

# print(len(data))
# pick_in = open('data1.pickle', 'wb')
# pickle.dump(data, pick_in)
# pick_in.close()

pick_in = open('cupInspectAOI\data1.pickle', 'rb')
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.9)

# model = SVC(C = 1,kernel = 'linear', gamma = 'auto')
# model.fit(xtrain, ytrain)

pick = open('cupInspectAOI\model.sav', 'rb')
# pickle.dump(model, pick)
model = pickle.load(pick)
pick.close()

prediction = model.predict(xtest)
accuracy = model.score(xtest, ytest)

categories = ['badProc', 'goodProc']

print("Accuracy: ", accuracy)

print('Predition is : ', categories[prediction[0]])

mycup = xtest[0].reshape(50, 50)
plt.imshow(mycup, cmap='gray')
plt.waitforbuttonpress()