# 最小二乘法
# 自变量与因变量之间必须有线性关系
# 对异常值非常敏感


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
# sklearn.metrics 用于模型评估，计算预测误差
from sklearn.metrics import mean_squared_error, r2_score # 均方误差
 
# 下载diabetes糖尿病数据集
diabetes = datasets.load_diabetes() 

# use only one feature
diabetes_X = diabetes.data[:, np.newaxis, 2]

# Split the data 最后20个用于测试
diabetes_X_train = diabetes_X[:-20] 
diabetes_X_test = diabetes_X[-20:]


# split the target
diabetes_y_train = diabetes.target[:-20] # target 结果
diabetes_y_test = diabetes.target[-20:]
# print(diabetes.target)
# print(diabetes_y_test)
# print(diabetes_y_train)

# create linear regression object
regr = linear_model.LinearRegression()

# train
regr.fit(diabetes_X_train, diabetes_y_train)

# prediction
diabetes_y_pred = regr.predict(diabetes_X_test)

# the coefficients
print('coefficients: \n', regr.coef_) # regr.coef_存放回归系数w(oumiga)

# the mean squared error 均方误差
print("Mean squared error: %.2f" 
    % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color='black') # 画点
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3) # 画预测线

plt.xticks(()) 
plt.yticks(())
plt.show()








