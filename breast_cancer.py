# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""



import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing,neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')


#defining features
novaLinha = ['id','Clump Thickness ','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
df.columns=novaLinha
primeiraLinha=[1000025,5,1,1,1,2,1,3,1,1,2]
df.loc[len(df)] = primeiraLinha

#cleaning data
df.replace('?',-99999,inplace=True)
df.drop(['id'],axis=1,inplace=True)


# defining X e Y
X = np.array(df.drop(['Class'], axis=1))
y = np.array(df['Class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#prediction
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[10,2,1,1,1,2,3,2,1]]) #transpondo
prediction = clf.predict(example_measures)
print(prediction)
