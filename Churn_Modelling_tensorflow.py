import tensorflow as tf
tf.__version__
#artifical Nural  Network
#implemnt to libraries
import numpy as np
import pandas as pd


#import the dataset
dataset = pd.read_csv(r"C:\Users\ICON\Downloads\Churn_Modelling.csv")
dataset.head()

X = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

X.shape
y.shape

X
y

dataset.head(1)
pip install scikit-learn

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
X

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

X

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)

X_train
X_train.shape
X_test.shape
y_train.shape
y_test.shape

#we are training the tensorflow

ann= tf.keras.models.Sequential()
ann

ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=6,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train,y_train,epochs=100)


y_pred = ann.predict(X_test)
y_pred = (y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)
