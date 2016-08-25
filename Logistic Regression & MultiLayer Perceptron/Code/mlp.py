import numpy as np
from sklearn.cross_validation import KFold
from sklearn.datasets import load_iris
import math
from sklearn.metrics import accuracy_score, confusion_matrix

iris = load_iris()


def sigmoid(x):
    output = 1 / (1 + np.exp(-x))
    return output

def derivative(op):
    op1 = op * (1 - op)
    return op1

X, y = iris.data, iris.target
y = y.reshape(y.shape[0], 1)
weight = np.zeros((X.shape[1], 1))
accuracy = 0.0
kf = KFold(X.shape[0], n_folds=20, shuffle=True)

for train, test in kf:
    X_Trn, X_Tst = X[train], X[test]
    Y_Trn, Y_Tst = y[train], y[test]


    for iter in range(0,100):

        layer0 = X_Tst
        layer1 = sigmoid(np.dot(layer0, weight))
        layer1_error = layer1 - Y_Tst
        layer1_del = layer1_error * derivative(layer1)
        synopse_0_derivative = np.dot(layer0.T, layer1_del)
        weight = weight - synopse_0_derivative

    predictedY = map(lambda x:math.ceil(x), layer1)
    Y_Tst = [x[0] for x in Y_Tst]

    accuracy += accuracy_score(Y_Tst, predictedY)


print "Multilayer Perceptron:"
print "\nAccuracy: "+str(round(accuracy,3)*10)+"%"