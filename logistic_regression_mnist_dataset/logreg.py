import numpy as np
import math

class LogisticRegression:
    def __init__(self, learning_rate, epoch, batch_size):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.batch_size = batch_size
        self.weight = None
        self.bias = None
        self.class_size = 10

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # preprocess - make train labels vertical - convert to one hot vector
        tmp = np.zeros((n_samples, self.class_size))
        tmp[np.arange(n_samples), y] = 1
        labels = tmp
        
        # init Weight and Bias with random values between 0, 1
        self.weight = np.random.rand(n_features, self.class_size)
        self.bias = np.random.rand(1, self.class_size)

        # Update
        for _ in range(self.epoch):
            for i in range(int(n_samples / self.batch_size)):

                session_x = X[self.batch_size * i:self.batch_size * (i + 1), :]
                session_y = labels[self.batch_size * i:self.batch_size * (i + 1), :]

                # logits = w.x + b
                logits = np.dot(session_x, self.weight) + self.bias

                # apply softmax and get predictions                
                y_pred = self.softmax(logits)

                # compute gradient descent and bias
                grad_desc =  np.dot(session_x.T, (y_pred - session_y))
                new_bias = np.sum(y_pred - session_y)

                # update weight & bias
                self.weight -= self.learning_rate * grad_desc
                self.bias -= self.learning_rate * new_bias

    def predict(self, X):
        predtest = self.softmax(np.dot(X, self.weight) + self.bias)
        values = predtest.max(axis=1).reshape(-1, 1)
        predtest[:] = np.where(predtest == values, 1, 0)
        return predtest
        
    def softmax(self, x):
        rows, cols = x.shape
        softmax = np.zeros((rows, cols))

        for i in range(rows):
            session = x[i, :]
            denominator = np.sum([(math.e ** k) for k in session])
            for j in range(cols):
                softmax[i, j] = (math.e ** session[j]) / denominator
        return softmax