import numpy as np
import matplotlib.pyplot as plt
from dataset import PerceptronData

class Perceptron(object):
    def __init__(self,dataset,lr = 0.01):
        self.dataset = dataset

        
        self.w = np.zeros((dataset.dims,1))    # 初始化w
        # self.w[0] = 1
        self.b = 0                              # 初始化b
        self.lr = lr                            # 初始化学习率

        self.all_w = [self.w]                   # 存储整个过程的 w
        self.all_b = [self.b]                   # 存储整个过程的 b
    
    def original_train(self):
        """原始的训练方法"""

        def train():
            for data,label in zip(self.dataset.data,self.dataset.label):
                data.reshape(1,-1)                                              # data --> shape: [1,2]
                if label * (np.dot(data,self.w) + self.b) <= 0:
                    self.w = self.w + self.lr * label * data.reshape(-1,1)
                    self.b = self.b + self.lr * label
                    return True
            return False

        while(train()):
            self.all_w.append(self.w)
            self.all_b.append(self.b)
        self.iter_num = len(self.all_w)

    def duality_train(self):
        """对偶学习方法"""
        alphas = np.zeros(self.dataset.nums)

        G = np.dot(self.dataset.data,np.transpose(self.dataset.data))   # Gram matrix, D*D^T
        def train():
            for i,label in enumerate(self.dataset.label):
                if label * (np.sum(alphas * self.dataset.label * G[i,:]) + self.b) <= 0:
                    alphas[i] = alphas[i] + self.lr
                    self.b = self.b + self.lr * label
                    return True
            return False
        
        while(train()):
            temp = alphas * self.dataset.label
            self.w = np.sum(temp.reshape(-1,1)*self.dataset.data,axis=0).reshape(-1,1)
            self.all_w.append(self.w)
            self.all_b.append(self.b)
        self.iter_num = len(self.all_w)

    def geney(self,x,w,b):
        """输入x/y,w,b,得到对应的y/x值"""
        if self.w[0,0] != 0:
            return -(w[1,0]*x + b)/self.w[0,0]
        else:
            return -(w[0,0] * x + b)/self.w[1,0]

    def show(self):
        """展示最终的结果"""
        point1_x,point2_x = min(self.dataset.data[:,0]),max(self.dataset.data[:,0])
        point1_y,point2_y = self.geney(point1_x,self.w,self.b),self.geney(point2_x,self.w,self.b)

        plt.figure()
        positive = np.where(self.dataset.label == 1)
        negtive = np.where(self.dataset.label == -1)
        plt.scatter(self.dataset.data[positive][:,0],self.dataset.data[positive][:,1],c='red',marker='.')
        plt.scatter(self.dataset.data[negtive][:,0],self.dataset.data[negtive][:,1],c='green',marker='.')
        plt.plot([point1_x,point2_x],[point1_y,point2_y],c='black')
        plt.show()
    
    def show_v2(self):
        """动态展示整个变化过程"""
        plt.figure()
        positive = np.where(self.dataset.label == 1)
        negtive = np.where(self.dataset.label == -1)
        point1_x,point2_x = min(self.dataset.data[:,0]),max(self.dataset.data[:,0])

        plt.scatter(self.dataset.data[positive][:,0],self.dataset.data[positive][:,1],c='red',marker='.')
        plt.scatter(self.dataset.data[negtive][:,0],self.dataset.data[negtive][:,1],c='green',marker='.')

        for i,(w,b) in enumerate(zip(self.all_w,self.all_b)):
            point1_y,point2_y = self.geney(point1_x,w,b),self.geney(point2_x,w,b)
            line, = plt.plot([point1_x,point2_x],[point1_y,point2_y],c='black')
            plt.title('Learning rate:{},w:[{:>.4f},{:>.4f}],b:{:>.3f}'.format(self.lr,w[0,0],w[1,0],b),fontsize=10)
            plt.xlabel('Iter nums:{}'.format(str(i)),fontsize=10)
            plt.pause(0.2)
            if i != self.iter_num - 1:
                line.remove()
        plt.show()
        plt.close()
        

if __name__ == '__main__':
    dataset = PerceptronData(200,2,0.1)
    perceptron = Perceptron(dataset,lr=0.1)
    perceptron.original_train()             # 原始的训练方法
    perceptron.show_v2()

    perceptron.duality_train()              # 使用对偶的训练方法
    perceptron.show_v2()
    