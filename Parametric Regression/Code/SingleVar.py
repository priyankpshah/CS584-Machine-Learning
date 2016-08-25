from array import array
import numpy as np
import matplotlib.pyplot as plot
from numpy.linalg import inv

data = np.loadtxt("svar-set4.dat")
data = np.loadtxt("svar-set4.dat", skiprows=5, dtype='str', delimiter=' ')

values = []

for i in data:
    values.append(filter(None, i))

value_float = np.array(values, dtype='float')

def calX(x,y,degree):
    m = (value_float.size / 2)
    x_degree = np.zeros((2*degree)+1)
    x_degree[0] = len(x)
    #Calculating A metrix values
    for i in range (1,len(x_degree+1)):
        x_degree[i] = np.sum(x**(i))
    #Calculating B matrix Value
    y_val = np.zeros(degree+1)
    y_val[0]=sum(y)
    for i in range(1,degree+1):
        y_val[i] = np.sum(y *(x**i))

    #Initializing matrix with zero
    A = np.matrix(np.zeros((degree+1,degree+1)))
    B = np.matrix(np.zeros(degree+1))
    #Filling A with data of X
    i=0
    for i in range(degree+1):
        for d in range (degree+1):
            A[i,d] = x_degree[d+i]

    #Filling B with value of Y

    B = y_val
    #B = np.matrix(B).T
    #Theta Calculation
    Theta = np.dot(inv(A), B)
    return Theta

#Calculation for Testing error
def Testing_error(testx,testy,Theta,degree):
    yt = np.zeros(len(testy))
    #Predicted Y
    Theta = np.matrix(Theta).T
    for i in range(len(yt)):
        for k in range(0,degree+1):
            yt[i] = yt[i] + np.sum(Theta[k]*(testx[i]**k))
            #print "i: " + str(i)
            #print "k: " + str(k)
            #print "yt: " + str(yt)

    yt_diff = np.matrix(yt).T - testy
    yt_diff_sqr = np.square(yt_diff)
    yt_fin = (np.mean(yt_diff_sqr))
    return yt_fin,yt

#Calculation for Training error
def Training_error(trainx,trainy,Theta,degree):
    yt = np.zeros(len(trainy))
    Theta = np.matrix(Theta).T
    #Predicted Y
    for i in range(len(trainx)):
        for k in range(0,degree+1):
            yt[i] = yt[i] + np.sum(Theta[k]*(trainx[i]**k))
    yt_diff = np.matrix(yt).T - trainy
    yt_diff_sqr = np.square(yt_diff)
    yt_fin = (np.mean(yt_diff_sqr))
    return yt_fin,yt

#Graph Plot for training and testing predicted Y
def plotgraph(testx,trainx,testy,trainy):

    plot.figure()
    plot.xlabel("Testing Data of X")
    plot.ylabel("Pradicted-Y")
    plot.title("PredictedY v/s Testing X - Dataset 4")
    plot.scatter(np.array(testx), np.array(testy))

    plot.figure()
    plot.xlabel("Training Data of X")
    plot.ylabel("Predicted-Y")
    plot.title("PredictedY v/s Training X - Dataset 4")
    plot.scatter(np.array(trainx), np.array(trainy))

    plot.xticks(())
    plot.yticks(())
    plot.show()

#Main
    #Kfold Input
KFold = input("Enter the number of Folds: ")
fold = value_float.shape[0]/KFold
i = 0
    #Polynomial degree Input
pol_degree = input("Enter the value of degree: ")
for i in range(Kfold):
    start_offset = ((fold) * i)
    end_offset = ((i + 1) * (fold)) - 1
    test_data = value_float[start_offset:end_offset+1, :]
    training_data = np.delete(value_float, range(start_offset, end_offset+1), 0)

    #dividing value from main matrix to test and train matrix x,y:
    value_testx = test_data[:, 0:-1]
    value_testy = test_data[:, 1:]
    value_trainx = training_data[:, 0:-1]
    value_trainy = training_data[:, 1:]
    value_x_whole = value_float[:, 0:-1]
    value_y_whole =  value_float[:, 1:]
    #Calling function for Theta Calculation
    Theta = calX(value_trainx,value_trainy,pol_degree)

    #Calling function for Testing Error Calculation
    ytest_fin,test_diff = Testing_error(value_testx,value_testy,Theta,pol_degree)

    #Calling function for Training Error Calculation
    ytrain_fin, train_diff = Training_error(value_trainx,value_trainy,Theta,pol_degree)

#output of value and graph
print "Testing Error:"
print np.min(ytest_fin)

print "Training Error:"
print np.min(ytrain_fin)
plotgraph(value_testx,value_trainx,test_diff,train_diff)
#plotgraph(value_x_whole,value_trainx,value_y_whole,train_diff)
