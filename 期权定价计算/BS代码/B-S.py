# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:43:19 2018

@author: 1
"""

import pandas as pd
import numpy as np
import math
from scipy.stats import norm

data=pd.read_csv('d://price1.csv',engine='python')
data=data[61:]
#数据读入

def european_call(r,sigma,T,S,K):
    d1=(math.log(S)-math.log(K)+(r+0.5*pow(sigma,2))*T)/(sigma*pow(T,0.5))
    d2=(math.log(S)-math.log(K)+(r-0.5*pow(sigma,2))*T)/(sigma*pow(T,0.5))
    V=S*norm.cdf(d1)-math.exp(-r*T)*K*norm.cdf(d2)
    return V
#欧式看涨期权定价

def european_put(r,sigma,T,S,K):
    d1=(math.log(S)-math.log(K)+(r+0.5*pow(sigma,2))*T)/(sigma*pow(T,0.5))
    d2=(math.log(S)-math.log(K)+(r-0.5*pow(sigma,2))*T)/(sigma*pow(T,0.5))
    V=-S*norm.cdf(-d1)+math.exp(-r*T)*K*norm.cdf(-d2)
    return V
#欧式看跌期权定价
data['theorical_price'] = data.apply(lambda data:european_call(0.03,data.sigma1,data.datedif,data.stock,3.66), axis = 1) 
#data['theorical_price'] = data.apply(lambda data:european_put(0.03,data.sigma1,data.datedif,data.stock,5.65), axis = 1)  
#函数入口

data.to_csv('d://g600038result.csv',encoding="utf_8_sig")



















#x=european_call(0.1,0.3,0.25,50,50)