from sklearn import datasets
import numpy as np
from collections import Counter
iris=datasets.load_iris

def euclidean_distance(x1,x2):
    np.sqrt(np.sum((x1-x2)**2))


#creating a KNN class

class KNN:
    def __init__(self,k=3): #init funtionin important
        self.k=k

    def fit(self,X, y):     #we need to make a functiom to fit training data to model
        self.X_train=X
        self.y_train=y

    def predict(self,X):      #we will loop the helper funtion to get a list of values
        predicted_labels= [self._predict(x) for x in X]
        return np.array(predicted_labels)


    def _predict(self,x):     #helper method to predict one value 
        #compute the distances
        distance= [euclidean_distance(x,x_train) for x_train in self.X_train]


        #get k nearest samples,labels
        k_indices= np.argsort(distance)[:self.k]
        k_neaarest_labels= [self.y_train[i] for i in k_indices]
        #majority vote, most common class labels
        most_common= Counter(k_neaarest_labels).most_common(1)
        return most_common[0][0]
