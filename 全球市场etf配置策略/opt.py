# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 21:52:26 2020

@author: 1
"""

#计算etf基金日度收益率
import datetime
import pandas as pd
import numpy as np
from pandas.core.frame import DataFrame
from sklearn.linear_model import LinearRegression
etf_data = pd.read_csv('data/data.csv',parse_dates=['Date'], dayfirst = True)
#etf_data = etf_data.dropna()

#etf日线收盘价数据转为周线
period_type = 'W'
etf_daily = etf_data.copy()
etf_daily.set_index('Date',inplace=True)
etf_week = etf_daily.resample(period_type,how='last')

#计算周收益率
columns = etf_week.columns
return_list=[]
for column in columns:
    etf_week[column+'_t-1'] = etf_week[column].shift(1)
    return_list.append(column+'_return')  
    etf_week[column+'_return'] = (etf_week[column] - etf_week[column+'_t-1'])/etf_week[column+'_t-1']

etf_return = etf_week[return_list]
etf_return = etf_return.dropna()

factor_data = pd.read_csv('data/F-F_Research_Data_Factors_weekly.CSV', parse_dates=['Date'], dayfirst = True) 
factor_data = factor_data[factor_data['Date'] > '2007-01-01']

def get_datetime_some_days_earlier(time):
    '''Get datetime some days earlier.'''
    return time + datetime.timedelta(days=2)

factor_data['Date'] = factor_data['Date'].apply(get_datetime_some_days_earlier)
factor_data.set_index('Date',inplace=True)
factor_data = factor_data/100

#合并数据、对每一个etf进行回归。
#较长的回溯期（≥120天）LT情况下收益和风险估计
merge_data = etf_return.join(factor_data)
begin_date = merge_data.index[0]
end_date = begin_date + datetime.timedelta(days=200)
regression_data = merge_data[(merge_data.index >= begin_date) & (merge_data.index <= end_date)]
regression_data = regression_data.dropna()

intercept=[]
b1=[]
b2=[]
b3=[]
return_columns = etf_return.columns
for column in return_columns:
    y_train = regression_data[column] - regression_data['RF']
    x_train = regression_data[['Mkt-RF','SMB','HML']]
    linreg = LinearRegression()  
    model=linreg.fit(x_train, y_train)  
    intercept.append(model.intercept_)
    b1.append(model.coef_[0])
    b2.append(model.coef_[1])
    b3.append(model.coef_[2])

model_coef = {"etf":return_columns,
   "B3":b1,
   "bs":b2,
   "bv":b3,
   "e":intercept}

model_coef=DataFrame(model_coef)

#对样本收益进行估计
outsample_data = merge_data[merge_data.index > end_date]
n = 0
expect_return_list=[]
for column in columns:
    outsample_data[column+'_expect_return'] =  model_coef.iloc[n,1]*outsample_data['Mkt-RF'] +  model_coef.iloc[n,2]*outsample_data['SMB']  +  model_coef.iloc[n,3]*outsample_data['HML'] + model_coef.iloc[n,4]
    merge_data[column+'_expect_return'] =  model_coef.iloc[n,1]*merge_data['Mkt-RF'] +  model_coef.iloc[n,2]*merge_data['SMB']  +  model_coef.iloc[n,3]*merge_data['HML'] + model_coef.iloc[n,4]
    expect_return_list.append(column+'_expect_return')
    n = n + 1
expect_return = outsample_data[expect_return_list]  

#Calculate trait volatility matrix
error_list = []
for column in columns:
    error_list.append(column+'_error')
    merge_data[column+'_error'] = merge_data[column+'_return'] - merge_data[column+'_expect_return'] 
error_data = merge_data[error_list]
error_var = error_data.var()
diag_matrix = np.eye(12)
for i in range(12):
    diag_matrix[i,i] = error_var[i]

#merge_data = merge_data.dropna()
#计算各etf的B(beta)系数
b=[]
return_columns = etf_return.columns
for column in return_columns:
    y_train = regression_data[column] - regression_data['RF']
    x_train = regression_data[['Mkt-RF']]
    linreg = LinearRegression()  
    model=linreg.fit(x_train, y_train)  
    b.append(model.coef_[0])
print('beta coef')
print(b)
print("")


#对样本风险协方差进行估计，同时进行优化组合。

# target beta optimization
import scipy.optimize as opt 
def obj_func(x,risk_cov,w_p,p):
    return  -1*(x.dot(p)  - 0.1*(x-w_p).dot(risk_cov).dot(x-w_p))

bnds=[]
for i in range(12):
    bnds.append((-0.2, 0.2))
    #bnds.append((-2, 2))#Range of parameters

w_p = np.ones(12)/12
x0 = np.ones(12)/12
outsample_data = outsample_data.dropna()

backtest_return = []

for i in range(len(outsample_data)):
#for i in range(len(outsample_data)-1):
#for i in range(1):
    #估计风险矩阵
    temp_data = merge_data.iloc[i:29+i,12:15]
    temp_return_data = merge_data.iloc[i:29+i,16:28]
    
    cov = np.array(temp_data.cov())
    B=np.array(model_coef[['B3','bs','bv']])
    risk_matrix = np.matmul(B, cov)
    risk_matrix = np.matmul(risk_matrix, B.T)+diag_matrix
 
    risk_return = temp_return_data.mean()
    
    cons=({'type': 'eq', 'fun': lambda x:  sum(x)-1}, {'type': 'eq', 'fun': lambda x: x.dot(b)-0.5})
    portfolio = opt.minimize(obj_func,x0,args=(risk_matrix,w_p,risk_return),bounds=bnds,constraints=cons)
    w_p = portfolio.x
    
    backtest_return.append(w_p.dot(np.array(merge_data.iloc[30+i,0:12])))
    
    print(portfolio)
    
print(backtest_return)
    
backtest_data = outsample_data[['SPY_return']]
#backtest_data = backtest_data.iloc[:,len(outsample_data)-1]
backtest_data['portfolio_return'] =  backtest_return

backtest_data = backtest_data.shift(1)
backtest_data.iloc[0,:]=0
backtest_nv = backtest_data + 1
backtest_nv = backtest_nv.cumprod()
backtest_nv.plot()

#Calculate risk metrics
def max_drawdown(timeseries):
    x=list(timeseries)
    max_=max(x)
    index=x.index(max_)
    min_=min(x[index:len(x)])
    max_drawdown=(max_-min_)/max_
    return(max_drawdown)

std = backtest_data['portfolio_return'].std()
skew = backtest_data['portfolio_return'].skew()
kurt = backtest_data['portfolio_return'].kurt()



#    print("each period expect_mean")
#    print(expect_return.iloc[i,:])
#    print("")
#    
#    print("each period risk_matrix")
#    print(risk_matrix)
#    print("")
    






