# -*- coding: utf-8 -*-
"""
Created on Mon May 21 20:26:03 2018

@author: xiexian
"""
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def nCr(n,r):
    f = math.factorial
    return f(n) / (f(r) * f(n-r))

def standardDeviation(rate):
     return numpy.std(rate)

class Option(object):
#每天的股价s0、波动率sigma、剩余时间t（年),要分的段数n不同，执行价格k相同
    def __init__(self,s0,sigma,r,t,n,k,type=''):
        self.s0=s0
        self.sigma=sigma
        self.r=r
        self.t=t
        self.n=n
        self.k=k
        self.type=type


    def Eu_price(self):
        time=float(self.t)/self.n
        u=math.exp(self.sigma*time**0.5)
        d=math.exp(-self.sigma*time**0.5)
        p=(math.exp(self.r*time)-d)/(u-d)
        payment=0
        for i in range(self.n+1):
            if self.type=='Call':#看涨期权
                pay=max(((self.s0*u**(self.n-2*i))-self.k),0)
            elif self.type=='Put':
                pay=max(self.k-((self.s0*u**(self.n-2*i))),0)
            ppay=pay*nCr(self.n,i)*((1-p)**i)*(p**(self.n-i))
            payment+=ppay
        payment=payment/math.exp(self.r*self.t)
        return payment

#××××××××××××××××××××××××××××××××××××××××××××××#


stock=pd.read_csv('price1.csv',engine='python')
timeList=np.array(stock['date'])
optionPrice=np.array(stock['option'])
stockPrice=np.array(stock['stock'])
rf=np.array(stock['r'])
sigmaList=np.array(stock['sigma'])


print 'sigmaList=',sigmaList
print 'timeList=',timeList
print '\n\noptionPrice=',optionPrice
print '\n\nstockPrice=',stockPrice
print 'len=',len(timeList),len(optionPrice),len(stockPrice)

print 'sigmaList=',sigmaList


curDay=62
dayNum=len(timeList)
predictPrice=[]
date=[]
realPrice=[]
realStockPrice=[]
while(curDay<=dayNum):
    s0=stockPrice[curDay-1]
    r=0.03
    k=3.66
    sigma=sigmaList[curDay-1]
    startTime = datetime.strptime(timeList[curDay-1],'%Y/%m/%d')
    endTime = datetime.strptime('2007-05-21','%Y-%m-%d')
    days=(endTime-startTime).days
    #print timeList[curDay-1]
    #print 'days=',days
    t=days/360.0
    n=days
    predictPrice.append(Option(s0,sigma,r,t,n,k,type='Call').Eu_price())
    date.append(timeList[curDay-1])
    realPrice.append(optionPrice[curDay-1])
    realStockPrice.append(stockPrice[curDay-1])
    curDay+=1
#print dayNum
print 'predictPrice=',predictPrice

return_={"date":date,
      "stockPrice":realStockPrice,
      "realPrice":realPrice,
      "predictPrice":predictPrice,
      }        
return_=pd.DataFrame(return_)
return_.to_csv('result.csv')


#××××××××××××××××××××××××××××××××××××××××××××××#
