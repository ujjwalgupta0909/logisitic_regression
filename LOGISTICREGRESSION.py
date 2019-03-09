# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 15:40:12 2018

@author: hp
"""

#import the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import the dataset
dataset=pd.read_csv("Social_Network_ads.csv")
features=dataset.iloc[:,2:4].values
labels=dataset.iloc[:,-1].values

#splitting
from sklearn.cross_validation import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
features_train=sc.fit_transform(features_train)
features_test=sc.transform(features_test)

#setting the model
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(features_train,labels_train)

#predicting the test set
labels_pred=classifier.predict(features_test)

#score
classifier.score(features_train,labels_train)
classifier.score(features_test,labels_test)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_pred)

#visualize(training set)
from matplotlib.colors import ListedColormap
x_set,y_set=features_train,labels_train
X1,X2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title("logistic regression")
plt.xlabel("age")
plt.ylabel("salary")
plt.legend()
plt.show()

#visualize(test set)
from matplotlib.colors import ListedColormap
x_set,y_set=features_test,labels_test
X1,X2=np.meshgrid(np.arange(start=x_set[:,0].min()-1,stop=x_set[:,0].max()+1,step=0.01),np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))

plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title("logistic regression")
plt.xlabel("age")
plt.ylabel("salary")
plt.legend()
plt.show()