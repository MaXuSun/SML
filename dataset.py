from operator import index
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

class Dataset(object):
    def __init__(self,nums,dims,errors=0):
        self.nums = nums
        self.dims = dims
        self.errors = errors

        self._data = None
        self._label = None
    
    def generate_data(self):
        pass
    def show(self):
        pass
    @property
    def data(self):
        return self._data

    @property
    def label(self):
        return self._label

class PerceptronData(Dataset):
    def __init__(self, *args,**knows):
        super().__init__(*args,**knows)
        self.generate_data()

    def generate_data(self):
        data = np.random.randn(self.nums,self.dims) + (np.random.rand(self.nums,self.dims)*self.errors)
        positive = np.where(data[:,0]+data[:,1] > 0)
        negtive = np.where(data[:,0]+data[:,1] <= 0)
        label = np.zeros(self.nums,dtype=int)
        label[positive] = 1
        label[negtive] = -1
        
        self._data = data
        self._label = label
    
    def show(self):
        plt.figure()
        positive = np.where(self._label == 1)
        negtive = np.where(self._label == -1)
        plt.scatter(self.data[positive][:,0],self.data[positive][:,1],c='red',marker='.')
        plt.scatter(self.data[negtive][:,0],self.data[negtive][:,1],c='green',marker='.')
        plt.show()

class KnnData(object):
    def __init__(self):
        iris = datasets.load_iris()
        self._data,self._label = iris.data,iris.target

        self.nums = self._data.shape[0]
        self.dims = self._data.shape[1]

    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label
    
    def generate_data(self):
        pass

    def show(self):
        assert self.dims == 2,'The numbers of the feature is bigger than two.'
        
if __name__ == '__main__':
    # dataset = PerceptronData(200,2,0.05)
    # dataset.show()

    dataset = KnnData()
    data = dataset.data
    print(np.median(data[:,2],axis=0))
    indexs = np.where(data[:,2]==np.median(data[:,2],axis=0))
    print(indexs)
    print(data[indexs])
    print(data[1:1])
    
