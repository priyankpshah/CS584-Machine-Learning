from __future__ import division
from sklearn.datasets import load_iris, fetch_mldata
from sklearn.cross_validation import KFold
import numpy as np


def indicator(class_fact, class_val):
    ind = []
    for label in class_fact:
        if label == class_val:
            ind.append(1)
        else:
            ind.append(0)
    return np.asarray(ind)

def Hypo(theta, X, thetas):
    den = exp_sum(thetas, X)
    hypo = np.exp(np.dot(X, theta))
    hypo /= den
    return hypo

def exp_sum(thetas, X):
    sum = 0
    m, n = np.shape(thetas)
    for i in range(n):
        sum += np.exp(np.dot(X, thetas[:, i]))
    return sum

def Find_Theta(X, Y, estimate, iterations):
    calssval = [0,1,2]
    x, y = np.shape(X)
    mul_theta = np.ones((y, len(calssval)))

    for j in range(iterations):
        for i, c in enumerate(calssval):

            theta = mul_theta[:, i]
            temp_hypo = Hypo(theta, X, mul_theta)
            ind = indicator(Y, c)
            theta_i = estimate * (np.sum((temp_hypo - ind).reshape(len(temp_hypo), 1) * X, axis=0))
            theta_i = theta_i.reshape(theta.shape)
            theta = theta - theta_i
            mul_theta[:, i] = theta
    return mul_theta

def confuide_mat(ytest, ypredict):
    cm = []
    clab = [0,1,2]
    for i in clab:
        tmp = [0] * len(clab)
        for j in range(len(ytest)):
            if ytest[j] == i and ytest[j] == ypredict[j]:
                tmp[clab.index(i)] += 1
            elif ytest[j] == i and ytest[j] != ypredict[j]:
                tmp[clab.index(ypredict[j])] += 1
        cm.append(tmp)
    return np.array(cm)

def predict(X_Test, thetas):

    Y_prediction = []

    thetas = thetas.T
    #print thetas
    for x in X_Test:

        h = -np.inf

        for i, theta in enumerate(thetas):

            h_hat = np.dot(x, theta)
            #print h_hat

            if h_hat > h:
                h = h_hat
                label = i
        Y_prediction.append(label)
    return Y_prediction

def confusion_mat(cm):

    precesion = np.zeros(2)
    recall = np.zeros(2)
    f1measure = np.zeros(2)
    accuracy = 0
    tot = np.sum(confusion_mat)
    for i in range(0,2):
        for j in range(0,2):
            precesion[i] += cm[j][i]
            recall[i] += cm[i][j]
            if(i==j):
                accuracy = accuracy + cm[i][j]
        precesion[i] = cm[i][i]/precesion[i]
        recall[i] = cm[i][i]/recall[i]
        f1measure[i] = ((2*precesion[i]*recall[i])/(precesion[i]+recall[i]))
    accuracy = float(accuracy)/tot


    return precision,recall,f_measure,accuracy

if __name__ == "__main__":

    mnist = fetch_mldata('MNIST original')
    X, Y = mnist.data / 255., mnist.target
    matrix = np.concatenate((X[Y == 0], X[Y == 1], X[Y == 2]), axis=0)
    y = np.concatenate((Y[Y == 0], Y[Y == 1], Y[Y == 2]), axis=0)

    kf = KFold(X.shape[0], n_folds=10, shuffle=True)

    accuracy = 0.0
    precision = np.zeros(3)
    recall = np.zeros(3)
    f_measure = np.zeros(3)

    for train,test in kf:

        X_Train, X_Test = X[train], X[test]
        Y_Train, Y_Test = Y[train], Y[test]
        thetas = Find_Theta(X_Train, Y_Train, 0.001, 2500)
        Y_Prediction = predict(X_Test, thetas)
        cm = confuide_mat(Y_Test, Y_Prediction)
        pre, rec, f1, acc = confusion_mat(cm)

        precision = np.add(precision, pre)
        recall = np.add(recall, rec)
        f_measure = np.add(f_measure, f1)
        accuracy = accuracy + acc

    precision = map(lambda x: x/10, precision)
    recall = map(lambda x: x/10, recall)
    f1measure = map(lambda x: x/10, f_measure)
    accuracy /= 10
    print " Confusion Matrix:"+ str(cm[0])
    print "\t\t\t\t  "+ str(cm[1])
    print "         Precesion:   Recall:    F-1 Measures"
    print "Class 0: " + str(round(precision[0],3))+"       "+str(round(recall[0],3))+"         "+str(round(f1measure[0],3))
    print "Class 1: " + str(round(precision[1],3))+"        "+str(round(recall[1],3))+"        "+str(round(f1measure[1],3))
    print "\nAccuracy: "+str(round(accuracy,3)*100)+ "%"
