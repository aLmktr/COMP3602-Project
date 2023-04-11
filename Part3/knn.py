import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import KFold

df = pd.read_csv("diagnosis.csv")

def euclidean_distance(point1, point2):
    """
    Compute the Euclidean distance between two data points.

    :param point1: An array representing the first data point.
    :param point2: An array representing the second data point.

    :return: The Euclidean distance between the two data points.
    """
    return sqrt(np.sum((point1 - point2) ** 2))

def knn(x_train, y_train, x_test, k):
    """
    Predict the class of a test point using the K-NN algorithm.

    :param x_train: An array of training data.
    :param y_train: An array of labels for the training data.
    :param x_test: An array representing the test point.
    :param k: The number of neighbors to consider.

    :return: The predicted class of the test point.
    """
    distances = np.zeros(x_train.shape[0])
    for i in range(x_train.shape[0]):
        distances[i] = euclidean_distance(x_train[i], x_test)
    indices = np.argsort(distances)
    class_votes = np.zeros(int(np.max(y_train)) + 1)
    for i in range(k):
        class_votes[int(y_train[indices[i]])] += 1
    return np.argmax(class_votes)

def cross_validate(x, y, k):
    """
    Perform ten-fold cross validation on the data using the K-NN algorithm.

    :param x: An array of data.
    :param y: An array of labels for the data.
    :param k: The number of neighbors to consider.

    :return: A list of accuracies for each fold.
    """
    kf = KFold(n_splits=10)
    accuracies = []
    for train_indices, test_indices in kf.split(x):
        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]
        y_test = y[test_indices]
        y_pred = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            y_pred[i] = knn(x_train, y_train, x_test[i], k)
        y_pred = y_pred.astype(int)
        y_test = y_test.astype(int)
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
    return accuracies

data = df.values
x = data[:,1:]
y = np.array([x[0] for x in data[:,:1]])
accuracies = cross_validate(x, y, 3)
print(accuracies)