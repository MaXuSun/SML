import numpy as np
from dataset import KnnData
from collections import defaultdict

def tree():
    return defaultdict(tree)

class KNN(object):
    def __init__(self):
        self.dataset = KnnData()
        self.kdTree = tree()

        self.dims = self.dataset.dims
        self.nums = self.dataset.nums

        self.count = 0
    def init_kdtree(self):
        self.construct_kdtree(self.kdTree,self.dataset.data,self.dataset.label,0)

    def construct_kdtree(self,root,data,label,lf):
        """构建KD树"""
        if len(data) == 0:            # 递归终止条件
            root=None
            return None

        if len(data.shape) == 1:        # 如果只有一个数据的时候，reshape成二维数据
            data.reshap(1,-1)
            label = (label,)

        median = np.median(data[:,lf],axis=0)           # 找到中值
        
        # 找到中值，比中值小，比中值大的下标
        index_median = np.where(data[:,lf]==median)
        index_small = np.where(data[:,lf] < median)
        index_big = np.where(data[:,lf] > median)

        # 找到中值，比中值小，比中值大的数据
        data_median,label_median = data[index_median],label[index_median]
        data_small,label_small = data[index_small],label[index_small]
        data_big,label_big = data[index_big],label[index_big]
        # print(data_small.shape,data_big.shape)

        if len(index_median[0]) == 0:                   # 如果中值是两个值得平均，就使用比中值小的所有值中最大的那个作为中值
            this_index = np.argmax(data_small[:,lf])

            root['data'] = data[this_index]
            root['label'] = label[this_index]

            data_small = np.delete(data_small,this_index,axis=0)
            label_small = np.delete(label_small,this_index,axis=0)
        else:                                           # 否则取中值中最中间的那个
            this_index = len(index_median[0])//2
            
            root['data'] = data_median[this_index]
            root['label'] = label_median[this_index]

            if len(data_big) == 0:                            # 如果big是空的，使用data_median更新big
                data_big = data_median[this_index+1:]
                label_big = label_median[this_index+1:]
            elif this_index == len(index_median)-1:                       # 如果big非空，data_median中big部分是空的，啥都不做
                pass
            else:                                       # 如果两个都不是空的，就合并
                data_big = np.concatenate([data_median[this_index+1:],data_big],axis=0)
                label_big = np.concatenate([label_median[this_index+1:],label_big],axis=0)

            if len(data_small) == 0:
                data_small = data_median[:this_index]
                label_small = label_median[:this_index]
            elif this_index == 0:
                pass
            else:
                data_small = np.concatenate([data_small,data_median[:this_index]],axis=0)
                label_small = np.concatenate([label_small,label_median[:this_index]],axis=0)
        root['level'] = lf

        lf = (lf+1)%self.dims
        # print(data_small,data_big)
        self.count += 1
        # print(root['data'],self.count,data_small.shape,data_big.shape)
        self.construct_kdtree(root['left'],data_small,label_small,lf)
        self.construct_kdtree(root['right'],data_big,label_big,lf)

if __name__ == '__main__':
    knn = KNN()
    knn.init_kdtree()