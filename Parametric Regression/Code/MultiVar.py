import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plot

data = np.loadtxt("mvar-set3.dat")
data = np.loadtxt("mvar-set3.dat", skiprows=5, dtype='str', delimiter=' ')

values = []
m = 0

def CalY():
    Theta = ((np.linalg.inv(value_zt * value_z_train)) * (value_zt * value_y_train))
    return Theta

def Y_Testing(ztest,Ytest,Theta):
    Y_predict = (ztest*Theta)
    ydiff = Y_predict - Ytest
    Y_predict_sqr = np.square(ydiff)
    error_test = np.mean(Y_predict_sqr)
    return error_test,Y_predict

def Y_Training(ztrain,ytrain,Theta):
    Y_predict= np.zeros(len(ytrain))
    Y_predict= ztrain*Theta
    ydiff = Y_predict - ytrain
    error_train_sqr = np.square(ydiff)
    error_train = (np.mean(error_train_sqr))
    return error_train,Y_predict

def Gradient_descent_train(x,y,Theta):
    x_trans = x.T
    n = np.shape(x)[1]
    iter= 20
    learning_rate = 0.001
    theta = np.ones((n,1),dtype='float')
    for i in range(iter):
        temp,predicted = Y_Training(x,y,theta)
        error = np.matrix(predicted).T - y

    grad_des = np.dot(x_trans, error) /x_trans.shape[0]
    theta = theta - learning_rate*grad_des
    return Theta

for i in data:
    values.append(filter(None, i))

value_float = np.array(values, dtype='float')
Kfold = input("Enter the number of folds: ")
fold = value_float.shape[0] / Kfold

i = 0
for i in range(10):
    start_offset = ((fold) * i)
    end_offset = ((i + 1) * (fold)) - 1

    test_data = value_float[start_offset:end_offset + 1, :]
    training_data = np.delete(value_float, range(start_offset, end_offset + 1), 0)

    val_z_test = test_data[:, 0:-1]
    value_y_test = test_data[:, -1:]

    val_z_train = training_data[:, 0:-1]
    value_y_train = training_data[:, -1:]

    array_one = np.ones((len(val_z_train), 1), dtype='float')
    value_z_train = np.c_[np.array(array_one), np.array(val_z_train)]

    array_one_test = np.ones((len(val_z_test), 1), dtype='float')
    val_z_test = np.c_[np.array(array_one_test), np.array(val_z_test)]

    value_zt = np.transpose(value_z_train)
    value_zt = np.asmatrix(value_zt, dtype='float')

    value_z_train = np.asmatrix(value_z_train, dtype='float')
    val_z_test = np.asmatrix(val_z_test,dtype='float')

    Theta = CalY()
    error_test,temp = Y_Testing(val_z_test,value_y_test,Theta)
    error_train,temp = Y_Training(value_z_train,value_y_train,Theta)

    #grad_result = Gradient_descent_train(value_z_train,value_y_train,Theta)

print "Testing Error:"
print error_test
print "Training Error:"
print error_train
print "Gradient Descent result:"
#print grad_result

