import numpy as np
from sklearn.metrics import confusion_matrix
import nD_2Class as twoclass
data = []



def collectdata():
    file = "spambase.data"
    data  = np.loadtxt(file,dtype=str,delimiter=",")
    np.random.shuffle(data)
    kfold = data.shape[0]/10
    features = data[:-(kfold),:-4]
    class_val = data[:-(kfold),-1]
    testdata = data[-(kfold):,:-4]
    test_val = data[-(kfold):,-1]
    features = np.array(features,dtype='float32')
    test_val = np.array(test_val,dtype='float32')
    for i in range(0,features.shape[0]):
        for j in range(0,features.shape[1]):
            if (features[i][j]!=0):
               features[i][j]=1
    return features,class_val,testdata,test_val

def calcalpha(features,classval):
    alpha = np.zeros((2,features.shape[1]))

    alpha[0] = np.sum(features[classval==np.unique((classval))[0]],axis=0,dtype='float')/(classval==np.unique(classval)[0]).sum()
    alpha[1] = np.sum(features[classval==np.unique((classval))[1]],axis=0,dtype='float')/(classval==np.unique(classval)[1]).sum()

    prior = np.zeros(2)
    prior[1] = classval.sum(axis=0)
    prior[0] = classval.shape[0] - prior[1]
    prior = prior/classval.shape[0]
    return alpha,prior

def CalcG(alpha,testData,prior):
    G = np.zeros((2, testData.shape[0]))

    for i in range(0, testData.shape[0]):
        G[0][i] =np.sum((testData[i] * np.log(alpha[0])) + ((1 - testData[i]) * np.log(1 - alpha[0])))
        G[1][i] =np.sum((testData[i] * np.log(alpha[1])) + ((1 - testData[i]) * np.log(1 - alpha[1])))
    G[0] += np.log(prior[0])
    G[1] += np.log(prior[1])
    return G

def yhat(G):
    ypredict = np.zeros(G.shape[1])
    for i in range(0,G.shape[1]):
        if(G[0][i]>G[1][i]):
            ypredict[i] = 0
        else:
            ypredict[i] = 1

    return ypredict


def confusionmatrix(ypredict, classval):
   temp = [0,1]
   cm= []
   for i in temp:
       tmp =[0]*len(temp)
       for j in range(len(classval)):
           if classval[j] == i and classval[j] == ypredict[j]:
               tmp[temp.index(i)] += 1
           elif classval[j] == i and classval[j] != ypredict[j]:
               tmp[temp.index(ypredict[j])] += 1
       cm.append(tmp)
   cm = np.array(cm)
   return cm

def precesionCall(cm,classval):
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
        precesion[i] = cm[i][i]/precesion[i]
        recall[i] = cm[i][i]/recall[i]
        f1measure[i] = ((2*precesion[i]*recall[i])/(precesion[i]+recall[i]))
    accuracy = float(accuracy)/classval.shape[0]

    return accuracy,precesion,recall,f1measure


if __name__ == '__main__':
    features, classval, testdata,test_val = collectdata()
    classval = np.array(classval,dtype='float')
    testdata = np.array(testdata,dtype='float')
    alpha,prior = calcalpha(features, classval)
    G = CalcG(alpha, testdata, prior)
    ypredict = yhat(G)
    cm = confusionmatrix(ypredict,test_val)
    accuracy,precesion,recall,f1measure = precesionCall(cm,test_val)

    print " Confusion Matrix:"+ str(cm[0])
    print "\t\t\t\t  "+ str(cm[1])
    print "         Precesion:   Recall:    F-1 Measures"
    print "Class 0: " + str(round(precesion[0],3))+"       "+str(round(recall[0],3))+"         "+str(round(f1measure[0],3))
    print "Class 1: " + str(round(precesion[1],3))+"        "+str(round(recall[1],3))+"        "+str(round(f1measure[1],3))
    print "\nAccuracy: "+str(round(accuracy,3)*100)+"%"

