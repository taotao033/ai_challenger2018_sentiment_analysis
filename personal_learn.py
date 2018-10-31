import numpy as np

a = np.array([[1,2,3],[3,4,5],[4,5,6]])
print(a)
print (a[...,1])   # 第2列元素
print (a[1,...])   # 第2行元素
print (a[...,1:])  # 第2列及剩下的所有元素



