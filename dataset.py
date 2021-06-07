import numpy as np
import matplotlib.pyplot as plt

class Dataset(object):
    def __init__(self,nums,dims,errors=0,norm=False,divide_equally=False):
        self.nums = nums
        self.dims = dims
        self.errors = errors
        self.norm = norm
        self.divide = divide_equally

        self._data = None
        self._label = None
    
    def generate_data(self):
        pass


    def normalize(self):
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
        print(positive)
        negtive = np.where(self._label == -1)
        print(self.data[positive,0].shape)
        plt.scatter(self.data[positive][:,0],self.data[positive][:,1],c='red',marker='.')
        plt.scatter(self.data[negtive][:,0],self.data[negtive][:,1],c='green',marker='.')
        plt.show()

if __name__ == '__main__':
    dataset = PerceptronData(200,2,0.05)
    dataset.show()
    
