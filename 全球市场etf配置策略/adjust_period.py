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
import matplotlib.pyplot as plt
#etf_data = etf_data.dropna()

#Calculate daily rate of return
etf_daily = etf_data.copy()
etf_daily.set_index('Date',inplace=True)
columns = etf_daily.columns
return_list=[]
for column in columns:
    etf_daily[column+'_t-1'] = etf_daily[column].shift(1)
    return_list.append(column+'_return')  
    etf_daily[column+'_return'] = (etf_daily[column] - etf_daily[column+'_t-1'])/etf_daily[column+'_t-1']
etf_return = etf_daily[return_list]
etf_return = etf_return.dropna()

#Read factor data
factor_data = pd.read_csv('data/F-F_Research_Data_Factors_daily.CSV', parse_dates=['Date'], dayfirst = True) 
factor_data = factor_data[factor_data['Date'] > '2007-01-01']
factor_data.set_index('Date',inplace=True)
factor_data = factor_data/100


#Combine the data and regress each etf.
merge_data = etf_return.join(factor_data)
merge_data = merge_data[merge_data.index <= '2008-03-01'] #before_crisis
#merge_data = merge_data[(merge_data.index >= '2008-01-01') & (merge_data.index <= '2011-01-01')] #before_crisis
#merge_data = merge_data[(merge_data.index >= '2011-01-01')] #before_crisis

#fama-french three-factor regression
def fama_regression(dataframe):
    intercept=[]
    b1=[]
    b2=[]
    b3=[]
    return_columns = etf_return.columns
    for column in return_columns:
        y_train = dataframe[column] - dataframe['RF']
        x_train = dataframe[['Mkt-RF','SMB','HML']]
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
    return model_coef

#capm regression
def capm_regression(dataframe):
    b=[]
    return_columns = etf_return.columns
    for column in return_columns:
        y_train = dataframe[column] - dataframe['RF']
        x_train = dataframe[['Mkt-RF']]
        linreg = LinearRegression()  
        model=linreg.fit(x_train, y_train)  
        b.append(model.coef_[0])
    return b
    
def get_expect_return(dataframe1,model_coef):
    n = 0
    dataframe = dataframe1.copy()
    expect_return_list=[]
    for column in columns:
        dataframe[column+'_expect_return'] =  model_coef.iloc[n,1]*dataframe['Mkt-RF'] +  model_coef.iloc[n,2]*dataframe['SMB']  +  model_coef.iloc[n,3]*dataframe['HML'] + model_coef.iloc[n,4]
        #merge_data[column+'_expect_return'] =  model_coef.iloc[n,1]*merge_data['Mkt-RF'] +  model_coef.iloc[n,2]*merge_data['SMB']  +  model_coef.iloc[n,3]*merge_data['HML'] + model_coef.iloc[n,4]
        expect_return_list.append(column+'_expect_return')
        n = n + 1
    #expect_return = dataframe[expect_return_list]
    return dataframe

def get_diag_matrix(dataframe):
    error_list = []
    for column in columns:
        error_list.append(column+'_error')
        dataframe[column+'_error'] = dataframe[column+'_return'] - dataframe[column+'_expect_return'] 
    error_data = dataframe[error_list]
    error_var = error_data.var()
    diag_matrix = np.eye(12)
    for i in range(12):
        diag_matrix[i,i] = error_var[i]
    return diag_matrix

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
#outsample_data = outsample_data.dropna()


j = 0
look_back_days = 60
look_back_days_list = [40,60,75]
beta_list = [-1.5,-0.5,0,0.5,1.5]
br= []
#for k in range(len(beta_list)):
for k in range(3):
    backtest_return = []
    look_back_days = look_back_days_list[k]
    for i in range(len(merge_data)-75):
    #for i in range(len(merge_data)-40):
    #for i in range(len(outsample_data)-1):
        #Estimated risk matrix
        temp_data = merge_data.iloc[i:look_back_days+i,12:15]
        
        regression_data = merge_data.iloc[i:look_back_days+i,0:16]
        model_coef = fama_regression(regression_data)
        b = capm_regression(regression_data)
        expect_return = get_expect_return(regression_data,model_coef)
        
        diag_matrix = get_diag_matrix(expect_return.copy())
        
        cov = np.array(temp_data.cov())
        B=np.array(model_coef[['B3','bs','bv']])
        risk_matrix = np.matmul(B, cov)
        risk_matrix = np.matmul(risk_matrix, B.T)+diag_matrix
     
        risk_return = expect_return.iloc[:,16:28]
        risk_return = np.array(risk_return.mean())
        if j%5 == 0:
            cons=({'type': 'eq', 'fun': lambda x:  sum(x)-1}, {'type': 'eq', 'fun': lambda x: x.dot(b)-0.5}) #adjust the param
            #cons=({'type': 'eq', 'fun': lambda x:  sum(x)-1}, {'type': 'eq', 'fun': lambda x: x.dot(b)+beta_list[k]}) #adjust the param
            portfolio = opt.minimize(obj_func,x0,args=(risk_matrix,w_p,risk_return),bounds=bnds,constraints=cons)
            w_p = portfolio.x
        
        backtest_return.append(w_p.dot(np.array(merge_data.iloc[75+i,0:12])))
        j = j+1
        #print(portfolio)
    br.append(backtest_return)
    
print(backtest_return)
m = len(merge_data)
backtest_data = merge_data.iloc[75:m,:]
backtest_data = backtest_data[['SPY_return']]

#backtest_data = outsample_data[['SPY_return']]
#backtest_data = backtest_data.iloc[:,len(outsample_data)-1]
backtest_data['portfolio_return_4060'] =  br[0]
backtest_data['portfolio_return_6090'] =  br[1]
backtest_data['portfolio_return_90120'] =  br[2]

backtest_data = backtest_data.shift(1)
backtest_data.iloc[0,:]=0
backtest_nv = backtest_data + 1
backtest_nv = backtest_nv.cumprod()
backtest_nv.plot()


def max_drawdown(timeseries):
    x=list(timeseries)
    min_=1
    max_drawdown=[]
    for i in range(1,len(x)):
        max_=max(x[0:i])
        index=x.index(max_)
        min_=min(x[index:len(x)])
        max_drawdown.append((max_-min_)/max_)
    return(max(max_drawdown))

def cal_indicator(dataframe):
    column = dataframe.columns
    print(column)
    mean = backtest_data[column].mean()*250
    var = backtest_data[column].var()*250
    skew = backtest_data[column].skew()
    kurt = backtest_data[column].kurt()
    #md = max_drawdown(np.array(backtest_nv[column]))
    sharpe = (mean-0.02)/(var**0.5)
    
    print('mean:'+str(mean))
    print('var:'+str(var))
    print('skew:'+str(skew))
    print('kurt:'+str(kurt))
    print('sharpe:'+str(sharpe))
    #print('md:'+str(md))

cal_indicator(backtest_data)

plt.show()
#
#fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6)) 
#ax0.hist(np.array(backtest_data['portfolio_return']),100,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
#plt.show()



