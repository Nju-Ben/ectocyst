# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 23:46:21 2020

@author: 1
"""


"""
数据预处理
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('data.csv')

#取前3000项为训练集，后1000项为测试集
train_data = data[0:2999]
test_data = data[2999:4000]
train_x = np.array(train_data.iloc[:,1:58])
test_x = np.array(test_data.iloc[:,1:58])

train_y = np.array(train_data.iloc[:,58:59])
test_y = np.array(test_data.iloc[:,58:59])

#同时标准化训练数据和测试数据
mean = train_x.mean(axis=0)
sigma = train_x.std(axis=0)

train_x = (train_x-mean)/sigma
test_x = (test_x-mean)/sigma

train_x = np.insert(train_x, 0, 1, axis=1) 
test_x = np.insert(test_x,0,1,axis=1)


"""
Logistic回归工具函数
"""

def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
def model(X, W):    
    return sigmoid(np.dot(X, W))    

def cost(X, y, W):
    left = np.multiply(y, np.log(model(X, W)))
    right = np.multiply(1 - y, np.log(1 - model(X, W)))
    return -np.sum(left + right) / (len(X))
 
def gradient(X, y, W):
    grad = np.zeros(W.shape)
    error = (model(X, W)- y)
    grad = np.dot(X.T, error)/len(X)
    return grad

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == 1:        return value > threshold
    elif type == 2:      return abs(value[-1]-value[-2]) < threshold
    elif type == 3:      return np.linalg.norm(value) < threshold

def map2class(list_):
    for i in range(len(list_)):
        if list_[i]>0.5:
            list_[i] = 1
        else:
            list_[i] = 0
    return list_
            
"""
Logistic回归类，包括训练、预测等流程
"""
class LogisticClassifer():
# 代价函数
    
    def __init__(self, stopType, thresh, alpha):
        self.stopType = stopType
        #初始化学习率、阈值、权重
        self.alpha = alpha
        self.thresh = thresh
        self.theta = np.matrix([1/58 for x in range(58)]).T
    
    #梯度下降求解训练模型
    def train(self,X,y):
        #梯度下降求解
        
        i = 0 # 迭代次数
        grad = np.zeros(self.theta.shape) # 计算的梯度
        costs = [cost(X, y, self.theta)] # 损失值
        
        while True:
            grad = gradient(X, y, self.theta)
            self.theta = self.theta - self.alpha*grad # 参数更新
            costs.append(cost(X, y, self.theta)) # 计算新的损失
            i += 1 
            
            # 何时停止
            if self.stopType == 1:       value = i
            elif self.stopType == 2:     value = costs
            elif self.stopType == 3:     value = grad
            if stopCriterion(self.stopType, value, self.thresh): break
        
        return self.theta, i-1, costs,
    
    def predict(self,X_test):
        y_test = model(X_test, self.theta)
        y_test = map2class(y_test)
        return y_test 
        
        #return self.theta, i-1, costs, grad, time.time() - init_time

epoch = 1000
alpha = 0.01

#调整参数学习率alpha
alpha_list = [0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
#fig, ax = plt.subplots(figsize=(12,4))

fig, ax = plt.subplots(figsize=(12,4))
for alpha_ in alpha_list:
    lr = LogisticClassifer(1,1000,alpha = alpha_)
    weight, iter_,loss = lr.train(train_x,train_y)
    predict_y = lr.predict(test_x)
    print(classification_report(test_y,predict_y))
    line = 'alpha = '+ str(alpha_)
    #ax.plot(np.arange(len(loss)), loss, 'r')
    ax.plot(np.arange(len(loss)), loss,label = line)

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Error')
    ax.set_title(' Error vs. Iteration with different learning-rate')
ax.legend()
plt.show()

#调sklearn的包
Lr = LogisticRegression(random_state=0,max_iter=200)
ml= Lr.fit(train_x,train_y)
predict_y_sk=ml.predict(test_x)
print(classification_report(test_y,predict_y_sk))
#



