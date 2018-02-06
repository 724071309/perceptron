# -*- coding: utf-8 -*-
from sklearn import svm, datasets, neighbors

iris = datasets.load_iris()

svc = svm.LinearSVC()
svc.fit(iris.data, iris.target) # 学习
print(svc.predict([[ 5.0, 3.0, 5.0, 2.0]]))


knn = neighbors.KNeighborsClassifier()
# 从已有数据中学习
knn.fit(iris.data, iris.target)
# 利用分类模型进行未知数据的预测（确定标签）
print(knn.predict([[5.0, 3.0, 5.0, 2.0]]))