# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 11:54:01 2021

@author: DELL
"""

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn


iris = datasets.load_iris()
df = pd.DataFrame(iris.data)
df['target'] = iris.target

X = df.drop('target', axis='columns')
Y = df.target

#split train test datasets
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)

model_score = model.score(x_test,y_test)
print(model_score)
y_predicted = model.predict(x_test)

cm = confusion_matrix(y_test,y_predicted)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

