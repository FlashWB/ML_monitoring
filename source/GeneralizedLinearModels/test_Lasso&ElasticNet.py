# Lasso and Elastic Net for Sparse Signals 对稀疏信号的处理

# Lasso回归 处理稀疏模型
# Lasso 使用L1范数  带有L1先验的正则项线性回归模型
# ElasticNet 弹性网络 使用L1,L2 范数作为先验正则训练的回归模型

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
###############################################################
# 生成42个值一样的随机数
np.random.seed(42) 

n_samples, n_features = 50, 200
X = np.random.randn(n_samples, n_features)     # 从正太分布中随机生成randn()
coef = 3 * np.random.randn(n_features)
# print(coef)
inds = np.arange(n_features)        # 生成0-199 arange()返回是对象np.arange[1,5]  返回array([1,2,3,4])
# print(inds)
np.random.shuffle(inds)             # 将序列所有元素打乱重新排序
# print(inds)
coef[inds[10:]] = 0                 # 每11个取一个值 稀疏
# print(coef)
y = np.dot(X, coef)                 # 点乘内积  np.multiply()相同位相乘

# add noise
y += 0.01 * np.random.normal(size=n_samples)

# split data
n_samples = X.shape[0]              # 0返回行数 1返回列数
# print(n_samples)
X_train, y_train = X[:n_samples // 2], y[:n_samples // 2]
X_test, y_test = X[n_samples // 2:], y[n_samples // 2:]

###############################################################
# lasso  
from sklearn.linear_model import Lasso
alpha = 0.1 # 惩罚项参数
lasso = Lasso(alpha=alpha)
# train
y_pred_lasso = lasso.fit(X_train, y_train).predict(X_test)
r2_score_lasso = r2_score(y_test, y_pred_lasso) # 决定系数 误差平方和/总平方和
print(lasso) #打印lasso参数
print("r² on test data: %f " %r2_score_lasso)


##############################################################
# ElasticNet 弹性网络 使用
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=alpha, l1_ratio=0.7) # alpha(α) l1_ratio(ρ)

y_pred_enet = enet.fit(X_train, y_train).predict(X_test)
r2_score_enet = r2_score(y_test, y_pred_enet)
print(enet)
print("r² on test data: %f " %r2_score_enet)

################################################################
# 画图 
plt.plot(lasso.coef_, color='lightgreen', linewidth=2,label='Lasso coefficients')
plt.plot(enet.coef_, color='red', linewidth=2,label='Elastic net coefficients')
plt.plot(coef, '--', color='navy', label='original coefficients') #原始值
plt.legend(loc='best') #设置显示比例 自适应
plt.title("Lasso R^2: %f, Elastic Net R^2: %f"
          % (r2_score_lasso, r2_score_enet))
plt.show()


