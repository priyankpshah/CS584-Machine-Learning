import numpy as np
import csv

class pca:
    def __init__(self):
        pass

    def fetchData(self):
        data = []
        readr = csv.reader(open('arrhythmia.data','rU'))

        for i in readr:
            data.append(i)
        return np.array(data)

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

    def findmean(self, value):

        mean = np.zeros((value.shape[0]))
        for i in range(0, value.shape[0]):
            mean[i] = np.sum(value[i])/value.shape[0]
        return mean

    def covar(self, mean, value):
        cov_mat = ((value - mean).T.dot((value - mean)))/value.shape[0]
        return cov_mat

    def eigenval(self, cov_mat):
        eig_val, eig_vec = np.linalg.eig(cov_mat)
        return eig_val,eig_vec

    def decrease_dimen(self, eigen_val, eigen_vec):

        eig_pairs = [(np.abs(eigen_val[i]),eigen_vec[:,i]) for i in range(0,len(eigen_val))]
        sorted_val = sorted(eig_pairs,key= lambda tup:tup[0])
        sorted_val.reverse()

        count = 0
        W = []
        for i in range(0,len(sorted_val)):
            if (sorted_val[i][0]>10):
                W.append(sorted_val[i][1])
                count = count + 1
        W = np.array(W)
        assert W.shape == (count,273), "Dimensions are not proper"

        return W

    def calc_svd(self, cov_mat):
        u, s ,v = np.linalg.svd(cov_mat)
        return u[:,:30]

    def calc_z(self,matrix_W, value):

        for i in range(0,value.shape[0]):
            z = np.dot(matrix_W,value.T)
        return z.T

if __name__ == '__main__':
    obj = pca()
    data = obj.fetchData()
    value,clable = obj.filterdata(data)
    mean = obj.findmean(value.T)
    cov_mat = obj.covar(mean,value)
    ureduce = obj.calc_svd(cov_mat)
    eigen_val,eigen_vec = obj.eigenval(cov_mat)
    matrix_W = obj.decrease_dimen(eigen_val,eigen_vec)
    z= obj.calc_z(ureduce.T,value)

'''Author: Priyank Shah
   CWID: A20344797 '''

