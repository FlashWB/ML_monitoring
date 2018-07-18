# Logistic Regression
# 交叉熵作代价函数
# SAGA algorithm 处理大数据集, 支持非平滑的L1正则 penalty="l1"
# 求解器基于平均随机梯度下降算法

import time
import numpy as np
import matplotlib.pyplot as plt

# api
from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression  # 从广义线性库调
from sklearn.model_selection import train_test_split # 随机划分训练和测试集,常用交叉验证
from sklearn.preprocessing import StandardScaler # 去均值和方差归一化
from sklearn.utils import check_random_state # np.random

t0 = time.time()
train_samples = 5000
mnist = fetch_mldata('MNIST original')
X = mnist.data.astype('float64') #(70000,784)的矩阵 
print(X)
y = mnist.target  # (70000,1)
print(y)
# 使用check_random_state 类初始化,使用random,permutation随机
random_state = check_random_state(0)
permutation = random_state.permutation(X.shape[0]) # 产生0-69999 共70000个随机数列表,用于随机排序X,y
X = X[permutation]  # 将 X 打乱 ,X[1].shape=(784,)
y = y[permutation]  #将 y 打乱 y.shape=(70000,) 
X = X.reshape((X.shape[0], -1)) # X.sahpe[0] X的行数

# 使用train_test_split随机划分训练集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000
)

scaler = StandardScaler() # 用于计算数据平均值和标准差
X_train = scaler.fit_transform(X_train)
# fit_tranform()先拟合数据再标准化,标准化使方差为1,均值为0,使得预测数据不会被某些维度过大的特征值而主导
X_test = scaler.transform(X_test) 

# Turn up tolerance for faster convergence 提高宽度加速收敛
clf = LogisticRegression(C=50. / train_samples, multi_class='multinomial',
    penalty='l1', solver='saga', tol=0.1)
clf.fit(X_train, y_train)

sparsity = np.mean(clf.coef_ == 0) * 100  # clf.coef_.shape=(10,784) ,对矩阵行列求均值
print(clf.coef_.shape) 
score = clf.score(X_test, y_test)
print(score.shape)

print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

coef = clf.coef_.copy()
print(coef.shape)
plt.figure(figsize=(10,5))
scale = np.abs(coef).max() # 最大绝对值
for i in range(10):
    l1_plot = plt.subplot(2, 5, i +1)
    l1_plot.imshow(coef[i].reshape(28,28), interpolation='nearest',
        cmap=plt.cm.RdBu, vmin=-scale, vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i ' % i)
plt.suptitle('Classification vector for...')

run_time = time.time()
print('Run in %.3f' % run_time)
plt.show()



