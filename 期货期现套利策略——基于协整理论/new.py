# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:25:00 2019

@author: 1
"""

import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as sts
from statsmodels.tsa.stattools import coint
import statsmodels.formula.api as smf 



import sys

def changedate(date):
    a=date.split(' ');
    b=date.split('/')
    date=b[0]+'/'+b[1]
    return date


stdout_backup = sys.stdout
log_file = open("message.log", "w")
# redirect print output to log file
sys.stdout = log_file

print ("Now all print info will be written to message.log")
# any command line that you will execute

stock=pd.read_csv('IC1906.csv',engine='python')
#stock=stock[0:5000]
stockdata=stock[['日期','收盘价(元)','南方中证','华夏中证']]
stockdata.rename(columns={'收盘价(元)':'IC1906'}, inplace = True)
#数据预处理
#处理掉空值
stockdata1=stockdata.dropna()
figure1=stockdata1[['日期','IC1906']]
figure2=stockdata1[['日期','南方中证']]
testdata=stockdata1[['日期','IC1906','南方中证','华夏中证']]

#处理到为零的值。
testdata=testdata[testdata['南方中证']!=0]

#testdata=testdata[testdata['华夏中证']!=0]

#南方中证etf价格换算，每手=价格*100
testdata['南方中证建模数据']=testdata['南方中证']*100
testdata['南方中证建模数据']=np.log(testdata['南方中证建模数据'])
testdata['南方中证收益率']=testdata['南方中证建模数据'].diff()/testdata['南方中证建模数据']
#南方中证etf对数一阶差分计算
testdata['南方中证delta']=testdata['南方中证建模数据'].diff()


#IC1906价格换算，价格=点数*200，同时按照论文思想取对数计算价差和一阶差分。
testdata['IC1906建模数据']=testdata['IC1906']*200
testdata['IC1906建模数据']=np.log(testdata['IC1906建模数据'])
testdata['IC1906收益率']=testdata['IC1906建模数据'].diff()/testdata['IC1906建模数据']
testdata['IC1906delta']=testdata['IC1906建模数据'].diff()
testdata1=testdata
testdata=testdata.dropna()
#计算相关系数
corr=testdata.corr() 

#ADF单位根检验
staIC1906 = sts.adfuller(testdata['IC1906建模数据'], 1)
staETF = sts.adfuller(testdata['南方中证建模数据'], 1)
staIC1906delta = sts.adfuller(testdata['IC1906delta'], 1)
staETFdelta = sts.adfuller(testdata['南方中证delta'], 1)


#python协整性检验 结果发现协整性检验通不过
cointtest=coint(testdata['IC1906建模数据'], testdata['南方中证建模数据'])
#cointtest2=coint(testdata['IC1906'], testdata['华夏中证'])

#进行OLS回归，确定对冲比例
est = smf.ols(formula='IC1906建模数据 ~ 南方中证建模数据', data=testdata).fit() 
print(est.summary())

#计算价差
testdata['spread']=testdata['IC1906建模数据']-1.0734*testdata['南方中证建模数据']

#中心化处理
testdata['mspread']=testdata['spread']-testdata['spread'].mean()
testdata['month']=testdata['日期'].apply(changedate)


#figure3=testdata[['mspread']].plot()

#staerror = sts.adfuller(testdata['spread'], 1)

#回归结果：LnIC1906=1.0734Ln南方中证+7.0594
#故对冲系数为1.0734

#交易部分
#价差序列标准差计算,设置仓位是否是空仓的flag信号
mean=testdata['spread'].mean()
sigma=testdata['mspread'].std()
flag=0

#timestamp=testdata['日期'].tolist()
#划分训练集和测试集
#data=np.array(testdata[['IC1906','南方中证','IC1906收益率','南方中证收益率','mspread']])

#rows=data.shape[0]
#columns=data.shape[1]
#train_data=data[0:int(rows*0.75)]
#test_data=data[int(rows*0.75):rows]


train_data=testdata[(testdata['month']!='2019/3') & (testdata['month']!='2019/4')]
test_data=testdata[(testdata['month']=='2019/3') |(testdata['month']=='2019/4')]
act_neg_flag=0#记录此前价差是大于均值（1）还是小于均值（0）
count=0#记录套利次数


lambda_=0.75
omega=2

#for i in range(len(train_data)):

for name,group in train_data.groupby('month'):
    timestamp=testdata['日期'].tolist()
    data=np.array(testdata[['IC1906','南方中证','IC1906收益率','南方中证收益率','mspread']])
    
    prices_etf=[]
    prices_IC=[]

    car=1
    portfolio_rate=[]
    print("套利月份"+name)
    for i in range(len(data)):
        mspread=round(data[i][4],6)
        etf_rate=round(data[i][3],6)
        IC1906rate=round(data[i][2],6)
        
        if flag==0:
            portfolio_rate.append(0)
            print("持仓状态：空仓 累计收益："+str(car)+ " 当期收益 0")
        else:
            if act_neg_flag==0:
                rate=1.0734*etf_rate-IC1906rate
                portfolio_rate.append(rate)
                car=car*(1+rate)
                print("持仓状态：1手IC1906空头，1.0734手南方中证多头 累计收益："+str(car*(1+rate))+ " 当期收益："+str(rate))
            else:
                rate=IC1906rate-1.0734*etf_rate
                portfolio_rate.append(rate)
                car=car*(1+rate)
                print("持仓状态：1手IC1906多头，1.0734手南方中证空头 累计收益："+str(car*(1+rate))+ " 当期收益："+str(rate))
        
        if mspread>=lambda_*sigma and mspread<omega*sigma and flag==0 and i!=len(train_data):
            prices_etf.append(data[i][1])
            prices_IC.append(data[i][0])
            act_neg_flag=1
            flag=1
            print("建仓，开一手IC1906多头，1.0734手南方中证空头,当前index："+str(i)+" IC1906价格："+str(data[i][0])+" 南方中证etf价格："+str(data[i][1]))
            
        if mspread<=-lambda_*sigma and mspread>-omega*sigma and flag==0 and i!=len(train_data):
            prices_etf.append(data[i][1])
            prices_IC.append(data[i][0])
            act_neg_flag=0
            flag=1
            print("建仓，开一手IC1906空头，1.0734手南方中证多头，当前index："+str(i)+" IC1906价格："+str(data[i][0])+" 南方中证etf价格："+str(data[i][1]))
            
            
        if mspread>=0 and flag==1 and act_neg_flag==0:
            prices_etf.append(data[i][1])
            prices_IC.append(data[i][0])
            count=count+1
            flag=0
            print("均值回归平仓，开一手IC1906多头，1.0734手南方中证空头，完成套利操作，当前index："+str(i)+" IC1906价格："+str(data[i][0])+" 南方中证etf价格："+str(data[i][1]))
            
        if mspread<=0 and flag==1 and act_neg_flag==1:
            prices_etf.append(data[i][1])
            prices_IC.append(data[i][0])
            count=count+1
            flag=0
            print("均值回归平仓，开一手IC1906空头，1.0734手南方中证多头，完成套利操作，当前index："+str(i)+" IC1906价格："+str(data[i][0])+" 南方中证etf价格："+str(data[i][1]))
            
        if mspread<=-omega*sigma and flag==1:
            prices_etf.append(data[i][1])
            prices_IC.append(data[i][0])
            flag=0
            print("触发止损平仓，开一手IC1906多头，1.0734手南方中证空头，当前index："+str(i)+" IC1906价格："+str(data[i][0])+" 南方中证etf价格："+str(data[i][1]))        
            
        if mspread>=omega*sigma and flag==1:
            prices_etf.append(data[i][1])
            prices_IC.append(data[i][0])
            flag=0
            print("触发止损平仓，开一手IC1906空头，1.0734手南方中证多头，当前index："+str(i)+" IC1906价格："+str(data[i][0])+" 南方中证etf价格："+str(data[i][1]))
        print(timestamp[i]+" 价格差:"+str(mspread)+" flag:"+str(flag)+" act_neg_flag"+str(act_neg_flag),end='')
    
    

#计算最大回撤函数
def max_drawdown(timeseries):
    x=list(timeseries)
    max_=max(x)
    index=x.index(max_)
    min_=min(x[index:len(x)])
    max_drawdown=(max_-min_)/max_
    return(max_drawdown)







log_file.close()
# restore the output to initial pattern
sys.stdout = stdout_backup

#testdata[testdata['南方中证']==0]['南方中证']='nan'
#testdata.fillna(method='ffill')



#画图
#figure2.plot()
#figure1.plot()


#stahuaxia = sts.adfuller(testdata['华夏中证'], 1)
#stahuaxiadelta = sts.adfuller(testdata['华夏中证delta'], 1)


#testdata['华夏中证']=testdata['华夏中证']*100
#testdata['华夏中证']=np.log(testdata['华夏中证'])
##南方中证etf对数一阶差分计算
#testdata['华夏中证delta']=testdata['华夏中证'].diff()





#cov=np.corrcoef(testdata)


#stockdata1.plot()