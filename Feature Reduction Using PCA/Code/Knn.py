# from PCA import value,clable
import numpy as np
import csv
import operator as op
import math
import time

class Knn:

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

        data = np.delete(data,1,1)
        Kfold = data.shape[0]/10
        np.random.shuffle(data)
        traindata = data[0:-Kfold,:]
        testdata = data[-Kfold:,:]
        return np.array(traindata,dtype='float32'),np.array(testdata,dtype='float32')
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
    obj = Knn()
    data = obj.fetchData()
    starttime = time.time()
    traindata,testdata = obj.filterdata(data)
    observations = []
    k = 5
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

