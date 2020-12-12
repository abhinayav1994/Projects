# Projects
@@ -0,0 +1,55 @@
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:31:01 2018

@author: LENOVO
"""

##identification of sepcies based on the features
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score,confusion_matrix

# importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier  
# import library for plotting
import matplotlib.pyplot as plt
data=pd.read_csv('Service.csv')

data_columns_list=list(data.columns)

data['Service']=data['Service'].map({'No':0, 
                         'Yes':1})
                         
features=list(set(data_columns_list)-set(['Service']))

x = data[features].values

y= data['Service'].values

train_x, test_x, train_y, test_y = train_test_split( x, y, test_size=0.3, random_state=0)

#data scaling
scaler = StandardScaler()

# Fit on training set only.
scaler.fit(train_x)

# Apply transform to both the training set and the test set.
train_x = scaler.transform(train_x)

test_x = scaler.transform(test_x)

# Storing the K nearest classfier
KNN_classifier = KNeighborsClassifier(n_neighbors=1)

KNN_classifier.fit(train_x,train_y)

prediction= KNN_classifier.predict(test_x)

confusion_matrix= confusion_matrix(test_y,prediction)

print("\t","Predicted values")

print("Original values","\n",confusion_matrix)

accuracy_score= accuracy_score(test_y, prediction)

print('Misclassified sample: %d' %(test_y!= prediction).sum())

Misclassified_sample=[]

##calculating the error for k values between 1 and 5
for i in range(1,5):
    knn= KNeighborsClassifier(n_neighbors= i)
    
    knn.fit(train_x, train_y)
    
    pred_i= knn.predict(test_x)
    
    Misclassified_sample.append((test_y!=pred_i).sum())
    
plt.figure(figsize=(5,4))

plt.plot(range(1,5,1), Misclassified_sample, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)

plt.title('Effect of K on misclassification')

plt.xlabel('Kvalue')

plt.ylabel('Misclassified samples')

plt.show()

#Therefore k can take values of 1 minimum for the classification of data
