# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 00:26:27 2019

@author: 1
"""

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
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import classification_report
data = pd.read_excel(r'共享单车数据十万条2.xls')
import statsmodels.formula.api as smf 

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

def getdate(time):
    date=time.split()[0]
    temp=date.split('/')
    month=temp[0]
    day=int(temp[1])
    if day<10:
        return int(str(month)+str(0)+str(day))
    else:
        return int(str(month)+str(day))

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
data['date']=data['开始时间'].apply(getdate)

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
    #if name<=1:
    if len(group)>5000:
        plot=plt.plot(group['开始维度'], group['开始经度'], '*')
#plot2=plt.plot(data['结束维度'], data['结束经度'], '*',color='blue',label='2')
plt.xlabel('jingdu')
plt.ylabel('weidu')
plt.show()
data=data[data['location']<=1]
correlation=data[['骑行时间','开始站点id','开始经度','开始维度',
                 '结束站点','结束经度','结束维度','车辆id',
                 '会员骑行时间','骑行路线类别','会员类型','startmonth','startday','hour','distance','location']].corr()

t=[]
c2=[]
c3=[]
tc2=[]
tc3=[]
to2=[]
to3=[]
for name1,group1 in data.groupby('date'):
    if int(name1)<900 and int(name1)>800:
    #if int(name1)>715:
        group2=group1[group1['location']==0]
        group3=group1[group1['location']==1]
        x2=group2.groupby(by='车辆id').agg({'骑行时间':sum}).reset_index()
        timec2=x2['骑行时间'].mean()
        tolc2=x2['骑行时间'].sum()
        
        tc2.append(timec2)
        
        x3=group3.groupby(by='车辆id').agg({'骑行时间':sum}).reset_index()
        timec3=x3['骑行时间'].mean()
        tolc3=x3['骑行时间'].sum()
        tc3.append(timec3)
        
        t.append(name1)
        
        count2=len(group2)
        count3=len(group3)
        c2.append(count2)
        c3.append(count3)
        
        to2.append(tolc2)
        to3.append(tolc3)
    
#p1 =plt.plot(t,c2)
#p2=plt.plot(t,c3)
p3=plt.plot(t,c2)
p4=plt.plot(t,c3)
plt.xlabel('date')
plt.ylabel('bike number')
plt.legend(['area0','area1'])
plt.show()


p5=plt.plot(t,tc2)
p6=plt.plot(t,tc3)
plt.xlabel('date')
plt.ylabel('average use time')
plt.legend(['area0','area1'])
plt.show()



df = {'date':t,
      'bikenum2':c2,
      'bikenum3':c3,
      'usetime2':tc2,
      'usetime3':tc3}
data_df = pd.DataFrame(df)
corr2=data_df[['bikenum2','usetime2']].corr()
est2 = smf.ols('usetime2 ~ bikenum2', data=data_df).fit()

corr3=data_df[['bikenum3','usetime3']].corr()
est3 = smf.ols('usetime3 ~ bikenum3', data=data_df).fit()
#print(est2.summary())
print(est3.summary())

































