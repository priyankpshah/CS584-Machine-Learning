import random

import numpy as np
from sklearn.metrics import metrics, confusion_matrix

data = np.loadtxt("data_banknote_authentication.txt",dtype='float', delimiter=',')
np.random.shuffle(data)
kfold = data.shape[0]/14
def findKfold():
    values_0 = []
    class_0 = []
    values_1 = []
    class_1 = []
    testdata = []

    for x in data[:-(kfold)]:
        line = [float(x) for x in x]
        if(line[-1] == 0.0):
            values_0.append(line[0:1])
            class_0.append(line[-1])
        else:
            values_1.append(line[0:1])
            class_0.append(line[-1])

    for x in data[-(kfold):]:
        line = [float(x) for x in x]
        testdata.append(line[0:1])

    values_0 = np.array(values_0)
    class_0 = np.array(class_0)
    values_1 = np.array(values_1)
    return values_0,values_1,class_0,np.array(testdata)

def meanCalculation(values_0,values_1):
    x_sum = np.zeros(1)
    x_mean = np.zeros(2)
    size_0 = values_0.shape[0]
    size_1 = values_1.shape[0]

    for j in range(0,size_0):
        x_sum = x_sum + values_0[j]
    x_mean[0] = float(x_sum) / size_0

    x_sum = np.zeros(1)
    for j in range(0,size_1):
        x_sum = x_sum + values_1[j]
    x_mean[1] = float(x_sum) / size_1

    return x_mean


def sigmaCalculation(mean, values_0, values_1):
    size_0 = values_0.shape[0]
    size_1 = values_1.shape[0]
    sigma = np.zeros(2)

    for i in range (0,size_0):
        sigma[0] = sigma[0] + ((values_0[i] - mean[0]) ** 2)
    sigma[0] = sigma[0]/size_0

    for i in range (0,size_1):
        sigma[1] = sigma[1] + ((values_1[i] - mean[1]) ** 2)
    sigma[1] = sigma[1]/size_1

    return sigma

def Gcalculation(mean, sigma,size_0,size_1,testdata):
    temp = data.shape[0]
    val = np.zeros(2)
    g = np.zeros((2,testdata.shape[0]))

    val[0] = -np.log(sigma[0])
    val[1] = -np.log(sigma[1])

    alpha = np.zeros(2)
    alpha[0] =  np.log(float(size_0)/temp)
    alpha[1] =  np.log(float(size_1)/temp)

    for i in range (0,testdata.shape[0]):
       g[0][i] = val[0]+ alpha[0] - (((testdata[i]-mean[0])**2)/(2*(sigma[0]**2)))
       g[1][i] = val[1]+ alpha[1] - (((testdata[i]-mean[1])**2)/(2*(sigma[1]**2)))

    return g

def accurator(G):
    difference = np.zeros(G.shape[1])
    yhat = np.zeros(G.shape[1])
    for i in range(0,G.shape[1]):
        difference[i] = G[0][i] - G[1][i]
        if (difference[i]<0):
            yhat[i]=1
        else:
            yhat[i]=0

    return yhat


def confusionmatrix(yhat, ytrue):
   temp = [0,1]
   cm= []
   ytrue = ytrue[-(kfold):]
   for i in temp:
       tmp =[0]*len(temp)
       for j in range(len(ytrue)):
           if ytrue[j] == i and ytrue[j] == yhat[j]:
               tmp[temp.index(i)] += 1
           elif ytrue[j] == i and ytrue[j] != yhat[j]:
               tmp[temp.index(yhat[j])] += 1
       cm.append(tmp)
   cm = np.array(cm)
   return cm


def precesionCall(cm):
    precesion = np.zeros(2)
    recall = np.zeros(2)
    f1measure = np.zeros(2)
    accuracy = 0
    for i in range(0,2):
        for j in range(0,2):
            precesion[i] += cm[j][i]
            recall[i] += cm[i][j]
            if(i==j):
                accuracy = accuracy + cm[i][j]
        if (precesion[i]!=0):
            precesion[i] = cm[i][i]/precesion[i]
            recall[i] = cm[i][i]/recall[i]
            f1measure[i] = ((2*precesion[i]*recall[i])/(precesion[i]+recall[i]))
        else:
            precesion[i] = 0
            recall[i] = 0
            f1measure[i] = 0
    accuracy = float(accuracy)/kfold

    return accuracy,precesion,recall,f1measure

def main():
    values_0,values_1,ytrue,testdata= findKfold()
    values = np.concatenate((values_0,values_1),axis=0)
    size_0 = values_0.shape[0]
    size_1 = values_1.shape[0]
    mean = meanCalculation(values_0,values_1)
    sigma = sigmaCalculation(mean,values_0,values_1)
    G = Gcalculation(mean,sigma,size_0,size_1,testdata)
    yhat = accurator(G)
    cm = confusionmatrix(yhat,ytrue)

    accuracy,precesion,recall,f1measure = precesionCall(cm)

    print "Confusion Matrix:"+ str(cm[0])
    print "\t\t\t\t  "+ str(cm[1])
    print "         Precesion:     Recall:    F-1 Measures"
    print "Class 0: " + str(round(precesion[0],3))+"           "+str(round(recall[0],3))+"       "+str(round(f1measure[0],3))
    print "Class 1: " + str(round(precesion[1],3))+"           "+str(round(recall[1],3))+"       "+str(round(f1measure[1],3))
    print "\nAccuracy:    "+str(accuracy)

if __name__ == '__main__' :
    main()
