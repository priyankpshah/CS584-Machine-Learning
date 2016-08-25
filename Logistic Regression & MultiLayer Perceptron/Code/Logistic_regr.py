from statsmodels.genmod.families.links import log
import numpy as np

class logistic_reg:


    def __init__(self):
        pass

    def fetchData(self):
        data = []
        file = open('iris.data.txt','rb')
        value = []
        class0 = []
        testdata = []
        testclass = []
        for i in file.readlines():
            data.append(i)

            kfold = len(data)/4
            np.random.shuffle(data)

        for i in range(0,len(data)-kfold):
            extract = data[i].split(',')
            if extract[4].split('\n')[0] == 'Iris-setosa':
                cur_data = (extract[0],extract[1],extract[2],extract[3])
                class0.append(0)
                value.append(cur_data)
            elif extract[4].split('\n')[0] == 'Iris-versicolor':
                cur_data = (extract[0],extract[1],extract[2],extract[3])
                class0.append(1)
                value.append(cur_data)

        for i in range(len(data)-kfold,len(data)):
            extract = data[i].split(',')
            if extract[4].split('\n')[0] == 'Iris-setosa':
                cur_data = (extract[0],extract[1],extract[2],extract[3])
                testdata.append(cur_data)
                testclass.append(0)
            elif extract[4].split('\n')[0] == 'Iris-versicolor':
                cur_data = (extract[0],extract[1],extract[2],extract[3])
                testdata.append(cur_data)
                testclass.append(1)
        print len(testdata) , len(value)
        return np.array(value,dtype='float32'),class0,np.array(testdata,dtype='float32'),np.array(testclass,dtype='float32')


    def cal_sigmoid(self,theta,val0):
        intermid = 1 + np.exp(-(np.dot(theta,val0.T)))
        sigmoid = 1 / intermid

        return np.array(sigmoid).T

    def cal_theta(self, th,val,classval):
        theta = th
        l_rate = 0.005
        classval = classval.T

        for i in range(500):
            sigmoid =  x.cal_sigmoid(theta,val)
            theta = theta - (l_rate * np.sum(((sigmoid - classval) * val),axis=0))

        print theta[0]

        return theta[0]

    def yhat(self, th_val, value):
        predicted_y = np.dot(th_val,value.T)
        com = 0
        for i in range(0,value.shape[0]):
            if (predicted_y[i]>com):
                predicted_y[i] = 1
            else:
                predicted_y[i] = 0

        return predicted_y

    def accuracy(self, ypredict, testclass):
       temp = [0,1]
       cm= []
       for i in temp:
           tmp =[0]*len(temp)
           for j in range(len(testclass)):
               if testclass[j] == i and testclass[j] == ypredict[j]:
                   tmp[temp.index(i)] += 1
               elif testclass[j] == i and testclass[j] != ypredict[j]:
                   tmp[temp.index(ypredict[j])] += 1
           cm.append(tmp)
       cm = np.array(cm)

       return cm


def precesionCall(cm, testclass):
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
    accuracy = float(accuracy)/testclass.shape[0]

    return accuracy,precesion,recall,f1measure

if __name__ == '__main__':

    x = logistic_reg()
    value,class0,testdata,testclass = x.fetchData()

    class0 = np.array(class0)[np.newaxis]
    theta = np.zeros(value.shape[1],dtype='float32')[np.newaxis]
    theta[:] = 0.02
    th_val = x.cal_theta(theta,value,class0)
    predicty = x.yhat(th_val,testdata)

    cm =  x.accuracy(predicty,testclass)
    accuracy,precesion,recall,f1measure = precesionCall(cm,testclass)

    print " Confusion Matrix:"+ str(cm[0])
    print "\t\t\t\t  "+ str(cm[1])
    print "         Precesion:   Recall:    F-1 Measures"
    print "Class 0: " + str(round(precesion[0],3))+"       "+str(round(recall[0],3))+"         "+str(round(f1measure[0],3))
    print "Class 1: " + str(round(precesion[1],3))+"        "+str(round(recall[1],3))+"        "+str(round(f1measure[1],3))
    print "\nAccuracy: "+str(round(accuracy,3)*100)+"%"
