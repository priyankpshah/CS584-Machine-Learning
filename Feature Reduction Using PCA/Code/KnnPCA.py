import numpy as np
import csv
import operator as op
import math
import time


class pca:
    def __init__(self):
        pass
    """ Retrive Data from the Database """
    def fetchData(self):
        data = []
        readr = csv.reader(open('arrhythmia.data','rU'))
        for i in readr:
            data.append(i)
        return np.array(data)
    """ Clean retrived Data"""
    def filterdata(self, data):
        temp = []
        for position,char in enumerate(data.T):
            if "?" in char:
                temp.append(position)
        for i in range(0,len(temp)):
            data = np.delete(data, 10, 1)
        clabel =  data[:,-1]
        data = np.delete(data,-1,1)
        data = np.delete(data,1,1)
        return np.array(data,dtype='float32'),np.array(clabel,dtype='float32')

    """Find the mean of each feature"""
    def findmean(self, value):

        mean = np.zeros((value.shape[0]))
        for i in range(0, value.shape[0]):
            mean[i] = np.sum(value[i])/value.shape[0]
        return mean
    '''Find Co-variance'''
    def covar(self, mean, value):
        cov_mat = ((value - mean).T.dot((value - mean)))/value.shape[0]
        return cov_mat
    ''' Find the eigen Value and Eigen Vetor from co-variance matrix'''
    def eigenval(self, cov_mat):
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        return eig_val,eig_vec
    '''Decrease the dimension with the consideration of highest value of eigen Value'''
    def decrease_dimen(self, eigen_val, eigen_vec):
        eig_pairs = [(np.abs(eigen_val[i]),eigen_vec[:,i]) for i in range(0,len(eigen_val))]
        sorted_val = sorted(eig_pairs,key= lambda tup:tup[0])
        sorted_val.reverse()
        count = 0
        W = []
        for i in range(0,len(sorted_val)):
            if (sorted_val[i][0]>30):
                W.append(sorted_val[i][1])
                count = count + 1
        W = np.array(W)
        assert W.shape == (count,273), "Dimensions are not proper"
        return W
    ''' Calculated SVD for checking purpose'''
    def calc_svd(self, cov_mat):
        u, s ,v = np.linalg.svd(cov_mat)
        return u[:,:30]
    '''from the reduced dimenson calculated matrix W, Find train Data and Test Data
    Data is shuffled'''
    def calc_z(self,matrix_W, value,clabel):
        for i in range(0,value.shape[0]):
            z = np.dot(matrix_W,value.T)
        z = z.real.T
        clabel = np.matrix(clabel).T
        z = np.concatenate((z,clabel),axis=1)
        # print z[:,-1]
        Kfold = z.shape[0]/10
        np.random.shuffle(z)
        traindata = z[0:-Kfold,:]
        testdata = z[-Kfold:,:]

        return np.array(traindata),np.array(testdata)
    """ Find the nearest neighbours based on the euclidian distnace"""
    def find_neighbour(self, traindata, param, k):
        distance = []
        neighbours = []
        size = len(param)-1

        for j in range(len(traindata)):
            edist = obj.euclidian_dist(param,traindata[j],size)
            distance.append((traindata[j],edist))
        distance.sort(key=op.itemgetter(1))
        for i in range(0,k):
            neighbours.append(distance[i][0])
        return neighbours
    """ Find the euclidian Distance between testdata example and one of the training data example"""
    def euclidian_dist(self, param, param1, size):
        distance = 0
        for i in range(size):
            distance = distance + np.power((param[i]-param1[i]),2)
        return math.sqrt(distance)
    """Once the distance is found, we will vote for the nearest neighbour"""
    def voteClass(self, neighbours):
        ind_vote = {}
        for i in range(len(neighbours)):
            vote = neighbours[i][-1]
            if vote in ind_vote:
                ind_vote[vote] += 1
            else:
                ind_vote[vote] = 1

        sort_vote = sorted(ind_vote.iteritems(),key= lambda elm: elm[1])
        return sort_vote[0][0]
    """ Check the voted class label and testdata lable and find the accuracy"""
    def find_accuracy(self, observations, testdata):
        true = 0
        for i in range(0,len(testdata)):
            if testdata[i][-1] == observations[i]:
                true += 1
        acc = (true/float(len(testdata)))

        return (acc*100)

if __name__ == '__main__':
    obj = pca()
    starttime = time.time()
    data = obj.fetchData()
    value,clable = obj.filterdata(data)
    mean = obj.findmean(value.T)
    cov_mat = obj.covar(mean,value)
    ureduce = obj.calc_svd(cov_mat)
    eigen_val,eigen_vec = obj.eigenval(cov_mat)

    matrix_W = obj.decrease_dimen(eigen_val,eigen_vec)
    traindata,testdata= obj.calc_z(ureduce.T,value,clable)

    observations = []
    '''value of nearest neighbours'''
    k = 3
    '''find neighbour for each test example and find the nearest neighbour for it'''
    for i in range(len(testdata)):
        neighbours = obj.find_neighbour(traindata,testdata[i],k)
        sort_result = obj.voteClass(neighbours)
        observations.append(sort_result)
    accuracy = obj.find_accuracy(observations,testdata)
    endtime = time.time()
    print "accuracy: " + str(round(accuracy,2)) + "%"
    print "Time: " + str(round(endtime - starttime,2)) + "Sec"

'''author: Priyank Shah
   CWID: A20344797'''
