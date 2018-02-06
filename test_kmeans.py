import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import svm

l1=[88,74,96,85]
l2=[92,99,95,94]
l3=[91,87,99,95]
l4=[78,99,97,81]
l5=[88,78,98,84]
l6=[100,95,100,92]
x=np.array([l1,l2,l3,l4,l5,l6])
kmeans=KMeans(n_clusters=2).fit(x)
p=kmeans.predict(x)
print(p)

