from KNN import KNN
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

iris= datasets.load_iris()
X,y= iris.data,iris.target

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1234)
clf= KNN(k=1)
clf.fit(X_train)

prediction= clf.predict(X_test)

accuracy= np.sum(prediction == y_test)/len(y_test)
print(accuracy)