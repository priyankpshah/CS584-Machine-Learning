import math
import numpy as np
import gmpy as gm

def collectdata():
    file = "spambase.data"
    data  = np.loadtxt(file,dtype=str,delimiter=",")
    np.random.shuffle(data)
    kfold = data.shape[0]/20
    doc_length = 100

    features = data[:-(kfold),:-4]
    class_val = data[:-(kfold),-1]

    testdata = data[-(kfold):,:-4]
    test_val = data[-(kfold):,-1]

    features = np.array(features,dtype='float32')
    testdata = np.array(testdata,dtype='float32')

    features*=doc_length
    testdata*=doc_length

    return np.array(features,dtype='float'),np.array(class_val,dtype='float'),np.array(testdata,dtype='float'),np.array(test_val,dtype='float')


def calcalpha(features, classval,wordcnt):
    alpha = np.zeros((2,features.shape[1]))
    epsilon = 1
    alpha[0] = (np.sum(features[classval==np.unique(classval)[0]],axis=0,dtype='float')+epsilon)/\
               ((wordcnt[classval==np.unique(classval)[0]]).sum()+len(np.unique(classval))*epsilon)
    alpha[1] = (np.sum(features[classval==np.unique((classval))[1]],axis=0,dtype='float')+epsilon)/\
               ((wordcnt[classval==np.unique(classval)[1]]).sum()+len(np.unique(classval))*epsilon)

    prior = np.zeros(2)
    prior[1] = classval.sum(axis=0)
    prior[0] = classval.shape[0] - prior[1]
    prior = prior/classval.shape[0]

    return alpha,prior


def CalcG(alpha, testfeatures, prior,tot_testval):
    G = np.zeros((2,testfeatures.shape[0]))

    for i in range (0,testfeatures.shape[0]):
        for j in range(0,testfeatures.shape[1]):
            ncr = gm.comb(int(tot_testval[i]),int(testfeatures[i][j]))
            G[0][i] = math.log(abs(ncr)) * (testfeatures[i][j] * math.log(abs(alpha[0][j]))) + (tot_testval[i] - testfeatures[i][j]) + math.log(abs(1-alpha[0][j]))
            G[1][i] = math.log(abs(ncr)) * (testfeatures[i][j] * math.log(abs(alpha[1][j]))) + (tot_testval[i] - testfeatures[i][j]) + math.log(abs(1-alpha[1][j]))
        G[0][i]+=np.log(prior[0])
        G[1][i]+=np.log(prior[1])
    return G


def yhat(G):
    ypredict = np.zeros(G.shape[1])

    for i in range(0,G.shape[1]):
        if(G[0][i]>G[1][i]):
            ypredict[i] = 0
        else:
            ypredict[i] = 1
        diff =  (G[0][i] - G[1][i])
    #print ypredict
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


def precesionCall(cm, classval):
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
    features,classval,testfeatures,testval = collectdata()
    tot_classval = np.sum(features,axis=1)
    tot_testval = np.sum(testfeatures,axis=1)
    alpha,prior = calcalpha(features, classval,tot_classval)
    G = CalcG(alpha, np.array(testfeatures), prior,tot_testval)
    ypredict = yhat(G)
    cm = confusionmatrix(ypredict,testval)
    accuracy,precesion,recall,f1measure = precesionCall(cm,testval)

    print " Confusion Matrix:"+ str(cm[0])
    print "\t\t\t\t  "+ str(cm[1])
    print "         Precesion:   Recall:    F-1 Measures"
    print "Class 0: " + str(round(precesion[0],3))+"       "+str(round(recall[0],3))+"         "+str(round(f1measure[0],3))
    print "Class 1: " + str(round(precesion[1],3))+"        "+str(round(recall[1],3))+"        "+str(round(f1measure[1],3))
    print "\nAccuracy: "+str(round(accuracy,3)*100)+"%"

