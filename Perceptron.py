#coding:utf8
import numpy as np
class Perceptron_cla(object):
    '''eta学习率
     n_iter训练权重向量次数
     w_神经分叉权重向量
     errors_用于记录神经元判断出错次数
     '''
    def __init__(self,eta,n_iter):
        self.eta = eta
        self.n_iter = n_iter
        pass
    def fit(self,x,y):
        '''输入培训数据，培训神经元，x输入样本向量，y对应样本分类
        x:shape[n_samples,n_features]
        x:[[1,2,3],[4,5,6]]     y:[1,-1]
        n_samples:  训练数据条目数
        n_features:3    含有数据的一维向量，用于表示一条训练条目
        '''
        self.w_ = np.zeros(1 + x.shape[1])  #初始化权重向量为0
        #self.w_ = [-0.4,2.0,1.82]
        self.errors_ = []       #加一是因为前面算法提到的w0，也就是步调函数阈值
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(x, y):
                update = self.eta*(target-self.predict(xi))
                self.w_[1:] += update *xi   #xi是向量
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
        print(self.w_)
        print(self.errors_)


    def net_input(self,x):  #z=w0*1+w1*x1+...wn*xn
        return np.dot(x,self.w_[1:]) + self.w_[0]

    def predict(self,x):
        return np.where(self.net_input(x)>= 0.0 ,1 ,-1)
