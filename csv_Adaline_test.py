#coding:utf8
file = 'test.csv'
#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from perceptron import Adaline_GD
from matplotlib.colors import ListedColormap


df = pd.read_csv(file,header=None)
#df.to_csv('test.csv',index=False,header=None)

y = df.loc[0:150,4].values
y = np.where(y =="Iris-setosa",-1,1)

x = df.loc[0:150,[0,2]].values
'''plt.scatter(x[:50,0],x[:50,1],color='red',marker='o',label='setosa')
plt.scatter(x[50:100,0],x[50:100,1],color='blue',marker='x',label='versicolor')
plt.xlabel(u'花瓣长度',fontproperties='SimHei')
plt.ylabel(u'花径长度',fontproperties='SimHei')
plt.legend(loc='upper left')'''

ada = Adaline_GD.AdalineGD(eta=0.0001, n_iter=50)
ada.fit(x,y)
'''plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
plt.xlabel('Epochs')
plt.ylabel(u'错误分类次数',fontproperties='SimHei')'''


def plot_decision_regions(x,y,classifier,resolution=0.02):
    marker = ('s','x','o','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])   #根据y不同结果分配不同颜色

    x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max()  #花径长度
    x2_min, x2_max = x[:, 1].min() - 1, x[:, 1].max()  #花瓣长度
    #print(x1_min,x1_max)
    #print(x2_min,x2_max)
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),  #扩展成二维矩阵
                          np.arange(x2_min,x2_max,resolution))
    #print(np.arange(x1_min,x1_max,resolution).shape)
    #print(np.arange(x1_min,x1_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z=z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y==cl,0],y=x[y==cl,1],alpha=0.8,c=cmap(idx),
                    marker=marker[idx],label=cl)



plot_decision_regions(x,y,ada,resolution=0.02)
plt.title('Adaline-Gradient descent')
plt.xlabel(u'花径长度',fontproperties='SimHei')
plt.ylabel(u'花瓣长度',fontproperties='SimHei')
plt.legend(loc='upper left')    #图例
plt.show()
#plt.close(0)

plt.plot(range(1,len(ada.cost_)+1),ada.cost_,marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-squard-error')
plt.show()