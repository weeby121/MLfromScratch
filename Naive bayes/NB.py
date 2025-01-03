import numpy as np

class NaiveBayes:

    def fit(self,X,y):
        n_samples , n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        #init mean,var,priors
        self._mean = np.zeros((n_classes , n_features), dtype=np.float64)
        self._var = np.zeros((n_classes , n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for c in self._classes:
            X_c=C[c==y]
            self._mean[c,:] = X_c.mean(axis=0)
            self._var[c,:] = X_c.var(axis=0)
            self._prior[c] = X_c.shape[0] / float(n_samples)



    def predict(self,X):
        y_pred = [self._predict(x) for x in X]

    def _predict(self,x):
        posterior = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            class_conditional = 

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var= self._var[class_idx]
        numerator = np.exp(-(x-mean)**2/(2*var))
        denominator = np.sqrt