# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 09:51:28 2019

@author: 1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import classification_report
data = pd.read_excel(r'共享单车数据十万条2.xls')


data['骑行路线类别'].replace(['Round Trip','One Way'] , [0,1] , inplace=True)
data['会员类型'].replace(['Walk-up','Monthly Pass','Flex Pass'] , [0,1,2] , inplace=True)

#获取日期和月份的函数
def getmonth(time):
    date=time.split()[0]
    temp=date.split('/')
    month=temp[0]
    return int(month)
    
def getday(time):
    date=time.split()[0]
    temp=date.split('/')
    day=temp[1]
    return int(day)

def gethour(time):
    timestamp=time.split()[1]
    temp=timestamp.split(':')
    hour=temp[0]
    return int(hour)


#该部分代码用于计算每次旅途
from math import sin, asin, cos, radians, fabs, sqrt 
EARTH_RADIUS=6371           # 地球平均半径，6371km 
def hav(theta):
    s = sin(theta / 2)
    return s * s 
def get_distance_hav(lat0, lng0, lat1, lng1):
    "用haversine公式计算球面两点间的距离。"
    # 经纬度转换成弧度
    lat0 = radians(lat0)
    lat1 = radians(lat1)
    lng0 = radians(lng0)
    lng1 = radians(lng1)
 
    dlng = fabs(lng0 - lng1)
    dlat = fabs(lat0 - lat1)
    h = hav(dlat) + cos(lat0) * cos(lat1) * hav(dlng)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h)) 
    return distance

#该部分代码用于计算每辆车的效率

#去除空值
data=data.dropna()    

#对时间进行处理
data['startmonth']=data['开始时间'].apply(getmonth)
data['startday']=data['开始时间'].apply(getday)
data['hour']=data['开始时间'].apply(gethour)

data['distance'] = data.apply(lambda x:get_distance_hav(x['开始维度'],x['开始经度'],x['结束维度'],x['结束经度']),axis=1)

#剔除了出发点与终点大于十公里的样本，这部分样本数量较少，仅有123个
data=data[data['distance']<=10]
#data=data[data['distance']!=0]


data1=data[['开始经度','开始维度']]
#经纬度图
#plot1=plt.plot(data['开始维度'], data['开始经度'], '*',color='red',label='1')
#plot2=plt.plot(data['结束维度'], data['结束经度'], '*',color='blue',label='2')
#plt.show()

#kmeans按距离聚类
from sklearn import cluster
k = 4
[centroid, label, inertia] = cluster.k_means(data1, k)
data['location']=label

for name,group in data.groupby('location'):
    if name<=1:
        plot=plt.plot(group['开始维度'], group['开始经度'], '*')
#plot2=plt.plot(data['结束维度'], data['结束经度'], '*',color='blue',label='2')
plt.xlabel('jingdu')
plt.ylabel('weidu')
plt.show()
data=data[data['location']<=1]
correlation=data[['骑行时间','开始站点id','开始经度','开始维度',
                 '结束站点','结束经度','结束维度','车辆id',
                 '会员骑行时间','骑行路线类别','会员类型','startmonth','startday','hour','distance','location']].corr()

train=data[['骑行时间','骑行路线类别','会员类型','startday','hour','distance','location']]

X_train,X_test,y_train,y_test = train_test_split(train[['骑行时间','骑行路线类别','会员类型','startday','hour','distance']],train['location'],test_size=0.1,random_state=0)
smo = SMOTE(random_state=40)
X_train, y_train = smo.fit_sample(X_train, y_train)
#Lr = LogisticRegression(C=1000,random_state=0)


Lr = LogisticRegression(C=1000,random_state=0,solver='newton-cg')
ml= Lr.fit(X_train,y_train)
y_score=ml.predict(X_train) 
#y_score=ml.predict(X_test)  
#y_test=np.array(y_test)
ml.coef_
#print(classification_report(y_test.astype('int'),y_score))
print(classification_report(y_train.astype('int'),y_score))

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X_train,y_train)
#x_new = lda.transform(X_test)
x_new = lda.transform(X_train)
mean=x_new.mean()
for i in range(len(x_new)):
    if x_new[i]>=mean:
        x_new[i]=1
    else:
        x_new[i]=0
#print(classification_report(y_test.astype('int'),x_new))
print(classification_report(y_train.astype('int'),x_new))
print(lda.coef_)

area0=data[data['location']==0]
area1=data[data['location']==1]

#可视化
#两地区租车时段差异。
plt.title('area0')
plt.xlabel('time/h')
plt.ylabel('count')
x0=area0['hour'].value_counts().index.tolist()
y0=area0['hour'].value_counts().tolist()
p1 =plt.bar(x0,y0)
plt.show()

plt.title('area1')
plt.xlabel('time/h')
plt.ylabel('count')
x1=area1['hour'].value_counts().index.tolist()
y1=area1['hour'].value_counts().tolist()
p2 =plt.bar(x1,y1)
plt.show()

#两地区会员差异
plt.title('area0')
plt.xlabel('VIP type')
plt.ylabel('portion')
x3=area0['会员类型'].value_counts().index.tolist()
count0=area0['会员类型'].value_counts().tolist()
y3=(np.array(count0)/np.array(count0).sum()).tolist()
p3 =plt.bar(x3,y3)
plt.xticks((0,1,2),('Walk-up','Monthly Pass','Flex Pass'))
plt.show()

plt.title('area1')
plt.xlabel('VIP type')
plt.ylabel('portion')
x4=area1['会员类型'].value_counts().index.tolist()
count1=area1['会员类型'].value_counts().tolist()
y4=(np.array(count1)/np.array(count1).sum()).tolist()
p4 =plt.bar(x4,y4)
plt.xticks((0,1,2),('Walk-up','Monthly Pass','Flex Pass'))
plt.show()

#两地区路线差异
plt.title('area0')
plt.xlabel('Route type')
plt.ylabel('portion')
x5=area0['骑行路线类别'].value_counts().index.tolist()
count2=area0['骑行路线类别'].value_counts().tolist()
y5=(np.array(count2)/np.array(count2).sum()).tolist()
p5 =plt.bar(x5,y5)
plt.xticks((0,1),('Round Trip','One Way'))
plt.show()

plt.title('area1')
plt.xlabel('Route type')
plt.ylabel('portion')
x6=area1['骑行路线类别'].value_counts().index.tolist()
count3=area1['骑行路线类别'].value_counts().tolist()
y6=(np.array(count3)/np.array(count3).sum()).tolist()
p6 =plt.bar(x6,y6)
plt.xticks((0,1),('Round Trip','One Way'))
plt.show()


#两地区距离差异
plt.title('area0')
plt.ylabel('distance')
d0=area0['distance'].mean()
d1=area1['distance'].mean()
p7 =plt.bar((0,1),(d0,d1))
plt.xticks((0,1),('area0','area1'))
plt.show()


#两地区骑行时间差异
plt.title('area0')
plt.ylabel('triptime')
d2=area0['骑行时间'].mean()
d3=area1['骑行时间'].mean()
p8 =plt.bar((0,1),(d2,d3))
plt.xticks((0,1),('area0','area1'))
plt.show()








#plt.title('area1')
#plt.xlabel('time/h')
#plt.ylabel('count')
#x4=area1['会员类型'].value_counts().index.tolist()
#count1=area1['会员类型'].value_counts().tolist()
#y4=(np.array(count1)/np.array(count1).sum()).tolist()
#p1 =plt.bar(x4,y4)
#plt.show()































