import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        n_samples , n_features = X.shape
        classes = np.unique(y)
        self.classes = classes
        n_classes = len(classes)
        self.y_mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.y_var =  np.zeros((n_classes, n_features), dtype=np.float64)
        self.prior_y = np.zeros((n_classes, 1), dtype=np.float64)

        for i, c in enumerate(self.classes):
            X_c = X[y==c]
            self.y_mean[i,:] = X_c.mean(axis=0)
            self.y_var[i,:] = X_c.var(axis=0)
            self.prior_y[i, :] = X_c.shape[0]/float(n_samples)
    def predict(self, X):
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        # HELPER Hn
        posteriors = []
        for i, c in enumerate(self.classes):
            class_cond_prob = np.sum(self._gaussian(i, x))
            posterior = class_cond_prob + self.prior_y[i]
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def _gaussian(self, i, x):
        mu = self.y_mean[i]
        var = self.y_var[i]
        exp = -1*(np.square(x-mu))/(2*var)
        return np.exp(exp)/np.sqrt(2*np.pi*var)