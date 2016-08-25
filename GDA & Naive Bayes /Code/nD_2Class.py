from array import array
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv
from sklearn.cross_validation import KFold
from sklearn.metrics import metrics, confusion_matrix, precision_recall_curve, average_precision_score


data = []
data = np.loadtxt("data_banknote_authentication.txt",dtype='float', delimiter=',')

def findKfold():

    values_0 = []
    class_0 = []
    values_1 = []
    class_1 = []

    np.random.shuffle(data)
    for x in data:
        line = [float(x) for x in x]
        if(line[-1] == 0.0):
            values_0.append(line[:-1])
            class_0.append(line[-1])
        else:
            values_1.append(line[:-1])
            class_1.append(line[-1])

    values_0 = np.array(values_0)
    class_0 = np.array(class_0)
    values_1 = np.array(values_1)
    class_1 = np.array(class_1)

    return values_0,values_1,class_0,class_1

def meanCalculation(values_0,values_1):
    x_sum = np.zeros(4)
    x_mean = np.zeros((2,4))
    size_0 = values_0.shape[0]
    size_1 = values_1.shape[0]

    for i in range(0,4):
        for j in range(0,size_0):
            x_sum[i] = x_sum[i] + values_0[j][i]
        x_mean[0,i] = x_sum[i] / size_0

    for i in range(0,4):
        for j in range(0,size_1):
            x_sum[i] = x_sum[i] + values_1[j][i]
        x_mean[1,i] = x_sum[i] / size_1

    return x_mean

def sigmaCalculation(x_mean,values_0,values_1):
    diff0 = np.zeros((values_0.shape))
    diff1 = np.zeros((values_1.shape))
    size_0 = values_0.shape[0]
    size_1 = values_1.shape[0]

    for i in range(0,4):
        for j in range(0,values_0.shape[0]):
            diff0[j][i] = values_0[j][i] - x_mean[0,i]

    sigma0 = np.dot(diff0.T,diff0)
    sigma0= sigma0/size_0

    for i in range(0,4):
        for j in range(0,values_1.shape[0]):
            diff1[j][i] = values_1[j][i] - x_mean[1,i]
    sigma1 = np.dot(diff1.T,diff1)
    sigma1 = sigma1/size_1
    return sigma0,sigma1,diff0,diff1

def Gcalculation(mean, sigma0,sigma1,diff,size_0,size_1):
    temp = data.shape[0]
    g = np.zeros((2,temp))
    val = np.zeros(2)
    val[0] = -np.log(np.linalg.det(sigma0))
    val[1] = -np.log(np.linalg.det(sigma1))

    alpha = np.zeros(2)
    alpha[0] =  np.log(float(size_0)/temp)
    alpha[1] =  np.log(float(size_1)/temp)

    for i in range (0,data.shape[0]):
       g[0][i] = val[0]+alpha[0] - (diff[i]*((np.linalg.inv(sigma0))*(np.matrix(diff[i]).T)))
       g[1][i] = val[1]+alpha[1] - (diff[i]*((np.linalg.inv(sigma1))*(np.matrix(diff[i]).T)))

    return g
def accurator(G):
    difference = np.zeros(G.shape[1])
    yhat = np.zeros(G.shape[1])

    for i in range(0,1372):
        difference[i] = G[0][i] - G[1][i]
        if (difference[i]<0):
            yhat[i]=1
        else:
            yhat[i]=0

    return yhat


def confusionmatrix(yhat, ytrue):
   temp = [0,1]
   cm= []
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


def main():
    values_0,values_1,class_0,class_1= findKfold()
    size_0 = values_0.shape[0]
    size_1 = values_1.shape[0]

    mean = meanCalculation(values_0,values_1)
    sigma0 , sigma1, diff0, diff1 =  sigmaCalculation(mean,values_0,values_1)
    diff= np.concatenate((diff0,diff1),axis=0)
    value_g = Gcalculation(mean,sigma0,sigma1,diff,size_0,size_1)
    ytrue = np.concatenate((class_0,class_1),axis=0)
    yhat = accurator(value_g)
    cm = confusionmatrix(yhat,ytrue)
    accuracy,precesion,recall,f1measure = precesionCall(cm,ytrue)
    print " Confusion Matrix:"+ str(cm[0])
    print "\t\t\t\t  "+ str(cm[1])
    print "         Precesion:          Recall:      F-1 Measures"
    print "Class 0: " + str(precesion[0])+"             "+str(recall[0])+"             "+str(f1measure[0])
    print "Class 1: " + str(precesion[1])+"             "+str(recall[1])+"             "+str(f1measure[1])
    print "\nAccuracy:    "+str(accuracy)

    precision,recall,thresold = precision_recall_curve(ytrue,yhat)
    average_precision = average_precision_score(ytrue,yhat)
    p_r_curve_fold ={}
    p_r_curve_fold.update({'precision':precision})
    p_r_curve_fold.update({'recall':recall})
    p_r_curve_fold.update({'average_precision':average_precision})


    recall = p_r_curve_fold['recall']

    precision = p_r_curve_fold['precision']
    average_precision = p_r_curve_fold['average_precision']

    plot.plot(recall, precision,
             label='Precision-recall curve (area = {1:0.2f})'
                   ''.format(0, average_precision))

    plot.xlim([0.0, 1.0])
    plot.ylim([0.0, 1.05])
    plot.xlabel('Recall')
    plot.ylabel('Precision')
    plot.title('Precision-Recall Curve')
    plot.legend(loc="lower right")
    plot.show()






if __name__ == '__main__':
    main()