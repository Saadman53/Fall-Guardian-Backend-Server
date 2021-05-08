import numpy as np 


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None


    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        y_ = np.where(y <= 0, -1, 1)
        X = X.reset_index()
        X = X.drop('index',axis = 1)
        
        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            
            for idx, x_i in X.iterrows():
                x_i = x_i.values
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
            


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        preds = np.sign(approx)
        preds = np.where(preds ==-1, 0, 1)
        return preds
    def svm_params(self):
        return self.w,self.b