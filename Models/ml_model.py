import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import pickle
import joblib
from joblib import dump

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler
dataset = pd.read_csv("diabetes.csv")
print(len (dataset))
print(dataset.head())
#Data Cleaning
zero_not_accepted=['Glucose','BloodPressure','SkinThickness','Insulin','BMI','Pregnancies','DiabetesPedigreeFunction','Age']
for column in zero_not_accepted:
    dataset[column]=dataset[column].replace(0,np.NaN)
    mean=int(dataset[column].mean(skipna=True))
    dataset[column]=dataset[column].replace(np.NaN,mean)
print(dataset["Glucose"])
#split dataset into train and test
X=dataset.iloc[:,0:8] #X_train = X_test
Y=dataset.iloc[:,8]    #Y_train
len(Y)
import math
math.sqrt(len(Y))
#define init model of knn
knn=KNeighborsClassifier(n_neighbors=3,p=2,metric='euclidean')
#fit model
knn.fit(X,Y)
#predicts the set results
Y_pred=knn.predict(X)  #Y_test
Y_pred
#evaluate ,odel mis classified data
cm=confusion_matrix(Y,Y_pred)
print(cm)
print(f1_score,(Y,Y_pred))
print(accuracy_score(Y,Y_pred))
aaa=knn.predict(np.array([5,88,66,21,23,24.4,0.342,30]).reshape(1, -1))
aaa
#Save Model With joblib
dump(knn, 'knn.joblib')

