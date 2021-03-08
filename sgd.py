import numpy as np
import random
import matplotlib.pyplot as plt
import operator
import math


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range, np.min(data), _range


class SGD():

    def __init__(self, learning_rate=.001, epochs=10, random_state=42, SGD_op='SGD1'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state
        self.sgd = SGD_op
        self.error = []
        self.W_list = []
        self.b_list = []

    def sigmoid(self, x):
        # return x
        return 1 / (1 + np.exp(-x))

    def feedforward_backpropagation(self, X, y, W, b):
        """
        X_nxm
        W_mX1
        """

        m = X.shape[0]

        A = self.sigmoid(np.dot(W.T, X) + b)
        # A = np.dot(W.T, X) + b

        loss = -1 / m * np.sum(y * np.log(A) + (1 - y) * np.log(1 - A))

        dw = 1 / m * np.dot(X, (A - y).T)
        db = 1 / m * np.sum(A - y)

        return loss, dw, db

    def fit(self, X, y, X_dev, y_dev, batch_size=32, ite_range=10000):
        self.cost_ = list()
        np.random.seed(self.random_state)

        class_num = []
        for item in y:
            if item not in class_num:
                class_num.append(item)
        for i in range(len(class_num)):
            self.W_list.append(np.random.random((X.shape[1], 1)))
            self.b_list.append(np.random.random())
        for i in range(self.epochs):
            print('Traing epoch:', i)
            halt = 0
            for ite in range(ite_range):
                if ite % 100 == 0:
                    self.error.append(self.get_error_num(X_dev, y_dev) / len(y_dev))
                dev = random.sample(list(zip(X, y)), batch_size)
                X_t, y_t = np.array([i1 for i1, i2 in dev]), np.array([i2 for i1, i2 in dev])
                for index in range(len(self.W_list)):
                    y_train = []
                    for item in y_t:
                        if item == index + 1:
                            y_train.append(1)
                        else:
                            y_train.append(0)
                    loss, dw, db = self.feedforward_backpropagation(X_t.T, np.array(y_train), self.W_list[index],
                                                                    self.b_list[index])
                    if db == 0:
                        halt += 1
                        # print('1')
                    if self.sgd == 'SGD1':
                        if halt == ite_range - 1:
                            print(halt)
                            print('Meet condition SGD1')
                            return self
                    if self.sgd == 'SGD2':
                        if halt > 100:
                            print('Meet condition SGD2')
                            return self
                    self.W_list[index] = self.W_list[index] - dw * self.learning_rate
                    self.b_list[index] = self.b_list[index] - db * self.learning_rate
        return self

    def predict_proba(self, X, y=None):
        negatives = self.sigmoid(np.dot(self.W.T, X.T) + self.b).T
        positives = 1 - negatives

        return np.hstack([positives, negatives])

    def predict(self, X, y=None, threshold=.5, index=0):
        probas = self.sigmoid(np.dot(self.W_list[index].T, X.T) + self.b_list[index]).T
        # probas = (np.dot(self.W.T, X.T) + self.b).T

        return np.where(probas >= threshold, 1, 0)

    def get_error_num(self, X_dev, y_dev):
        result_list = []
        for i in range(len(self.W_list)):
            result_list.append(self.predict(X_dev, index=i))
        reslut_array = np.array(result_list)
        return np.count_nonzero(np.dot(reslut_array.T, np.array([1, 2, 3])) - y_dev)


import pandas as pd
import sgd
df1 = pd.read_csv("wine_train.csv")
df2 = pd.read_csv("wine_test(1).csv")

X = df1.iloc[:, :-1].values
y = df1.iloc[:, -1].values
X_dev = df2.iloc[:, :-1].values
y_dev = df2.iloc[:, -1].values
result=[]
sgd_list = []
sdg1 = sgd.SGD(SGD_op='SGD1')
sdg1.fit(X, y, X_dev, y_dev)
print('sgd1',sdg1.error[-1])
sdg1 = sgd.SGD(SGD_op='SGD2')
sdg1.fit(X, y, X_dev, y_dev)
print('sgd2',sdg1.error[-1])

# for i in range(10):
#     sgd_list.append(sgd.SGD(SGD_op='SGD1'))
#     sgd_list[-1].fit(X, y, X_dev, y_dev)
#     result.append(sgd_list[-1].error)

