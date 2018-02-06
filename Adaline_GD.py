#coding:utf8
import numpy as np

class AdalineGD(object):

    def __init__(self,eta,n_iter):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self,x,y):
        self.w_ = np.zeros(1 + x.shape[1])
        self.cost_ = []    #成本向量

        for i in range(self.n_iter):
            output = self.net_input(x)  #Z
            errors = (y - output)   #(y-z)
            self.w_[1:] += self.eta * x.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)

    def net_input(self,x):  #z=w0*1+w1*x1+...wn*xn
        return np.dot(x,self.w_[1:]) + self.w_[0]

    def activation(self,x):
        return self.net_input(x)

    def predict(self,x):
        return np.where(self.activation(x) >= 0, 1, -1 )