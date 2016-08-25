import numpy as np
from sklearn.metrics import metrics, confusion_matrix
import math

data = []

def Fetchdata():
    file = open('iris.data.txt','rb')

    value0 = [] ; value1 = [] ; value2= []
    class0 = []
    testdata = []
    testclass = []
    for i in file.readlines():
       data.append(i)
    kfold = len(data)/5
    np.random.shuffle(data)
    for i in range(0,len(data)-kfold):
        extract = data[i].split(',')
        if extract[4].split('\n')[0] == 'Iris-setosa':
            cur_data = (extract[0],extract[1],extract[2],extract[3])
            class0.append(0)
            value0.append(cur_data)
        elif extract[4].split('\n')[0] == 'Iris-versicolor':
            cur_data = (extract[0],extract[1],extract[2],extract[3])
            class0.append(1)
            value1.append(cur_data)
        else:
            cur_data = (extract[0],extract[1],extract[2],extract[3])
            value2.append(cur_data)
            class0.append(2)

    for i in range(len(data)-kfold,len(data)):
        extract = data[i].split(',')
        cur_data = (extract[0],extract[1],extract[2],extract[3])
        testdata.append(cur_data)
        if extract[4].split('\n')[0] == 'Iris-setosa':
            testclass.append(0)
        elif extract[4].split('\n')[0] == 'Iris-versicolor':
            testclass.append(1)
        else:
            testclass.append(2)

    return np.array(value0,dtype='float32'),np.array(value1,dtype='float32'),np.array(value2,dtype='float32'),class0,np.array(testdata,dtype='float32'),np.array(testclass,dtype='float32')

def meanCalculation(value0,value1,value2):
    x_sum = np.zeros(4)
    x_mean = np.zeros((3,4))
    size = len(value0)

    for i in range(0,4):
        for j in range(0,value0.shape[0]):
            x_sum[i] = x_sum[i] + value0[j][i]
        x_mean[0,i] = x_sum[i] / size

    x_sum = np.zeros(4)
    for i in range(0,4):
        for j in range(0,value1.shape[0]):
            x_sum[i] = x_sum[i] + value1[j][i]
        x_mean[1,i] = x_sum[i] / size

    x_sum = np.zeros(4)
    for i in range(0,4):
        for j in range(0,value2.shape[0]):
            x_sum[i] = x_sum[i] + value2[j][i]
        x_mean[2,i] = x_sum[i] / size
    return np.array(x_mean,dtype='float32')


def sigmaCalculation(mean, value0, value1,value2):
    diff0 = np.zeros(value0.shape)
    diff1 = np.zeros(value1.shape)
    diff2 = np.zeros(value2.shape)
    size = value0.shape[0]

    for i in range(0,4):
        for j in range(0,value0.shape[0]):
            diff0[j][i] = value0[j][i] - mean[0][i]
    sigma0 = np.dot(diff0.T,diff0)
    sigma0= sigma0/size

    for i in range(0,4):
        for j in range(0,value1.shape[0]):
            diff1[j][i] = value1[j][i] - mean[1][i]
    sigma1 = np.dot(diff1.T,diff1)
    sigma1 = sigma1/size

    for i in range(0,4):
        for j in range(0,value2.shape[0]):
            diff2[j][i] = value2[j][i] - mean[2][i]
    sigma2 = np.dot(diff2.T,diff2)
    sigma2 = sigma2/size

    return sigma0,sigma1,sigma2


def Gcalculation(sigma0, sigma1, sigma2, testdata,mean):
    g = np.zeros((3,30))
    val = np.zeros(3)

    diff0 = testdata - mean[0]
    diff1 = testdata - mean[1]
    diff2 = testdata - mean[2]
    print type(diff0), diff0.shape , sigma0.shape

    val[0] = - math.log(np.linalg.det(sigma0))
    val[1] = - math.log(np.linalg.det(sigma1))
    val[2] = -math.log(np.linalg.det(sigma2))
    alpha =  math.log(float(50)/150)
    for i in range (0,30):
       g[0][i] = val[0]- (np.matrix(diff0[i])*np.matrix(np.linalg.inv(sigma0))*(np.matrix(diff0[i]).T))+alpha
       g[1][i] = val[1]- (np.matrix(diff1[i])*np.matrix(np.linalg.inv(sigma1))*(np.matrix(diff1[i]).T))+alpha
       g[2][i] = val[2]- (np.matrix(diff2[i])*np.matrix(np.linalg.inv(sigma2))*(np.matrix(diff2[i]).T))+alpha

    return g


def accurator(G, ytrue):

    yhat = np.zeros(G.shape[1])

    for i in range(0,G.shape[1]):
        if (G[0][i]>G[1][i]):
            if(G[0][i]>G[2][i]):
                yhat[i] = 0
            else:
                yhat[i] = 2
        elif (G[1][i]>G[2][i]):
            yhat[i] = 1
        else:
            yhat[i] = 2

    return yhat


def confusionmatrix(yhat,ytrue):
   temp = [0,1,2]
   cm= []
   print yhat.shape

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
    precesion = np.zeros(3)
    recall = np.zeros(3)
    f1measure = np.zeros(3)
    accuracy = 0
    for i in range(0,3):
        for j in range(0,3):
            precesion[i] += cm[j][i]
            recall[i] += cm[i][j]
            if(i==j):
                accuracy = accuracy + cm[i][j]
        precesion[i] = cm[i][i]/precesion[i]
        recall[i] = cm[i][i]/recall[i]
        f1measure[i] = ((2*precesion[i]*recall[i])/(precesion[i]+recall[i]))
    accuracy = float(accuracy)/30

    return accuracy,precesion,recall,f1measure

if __name__ == '__main__':

  value0,value1, value2,class0,testdata,testclass = Fetchdata()
  mean = meanCalculation(value0,value1,value2)
  sigma0,sigma1,sigma2=  sigmaCalculation(mean,value0,value1,value2)
  value_g = Gcalculation(sigma0,sigma1,sigma2,testdata,mean)
  yhat = accurator(value_g,class0)
  cm = confusionmatrix(yhat,np.array(testclass))
  accuracy,precesion,recall,f1measure = precesionCall(cm)
  print " Confusion Matrix:"+ str(cm[0])
  print "\t\t\t\t  "+ str(cm[1])
  print "\t\t\t\t  "+ str(cm[2])
  print "         Precesion:          Recall:      F-1 Measures"
  print "Class 0: " + str(precesion[0])+"      "+str(recall[0])+"        "+str(f1measure[0])
  print "Class 1: " + str(precesion[1])+"               "+str(recall[1])+"         "+str(f1measure[1])
  print "Class 2: " + str(precesion[2])+"      "+str(recall[2])+"         "+str(f1measure[2])
  print "\nAccuracy:    "+str(accuracy)

