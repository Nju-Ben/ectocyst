# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 15:52:11 2019

@author: 1
"""

#matplotlib inline

import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
import pickle
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import functools
import warnings

warnings.filterwarnings('ignore')

def timeStamp(timeNum): 
    
    timestamp = float(timeNum/1000000)
    otherStyleTime=datetime.fromtimestamp(timestamp)
    
    return otherStyleTime

#def trade(trade_data, book_data, s = 0.01, j = 0.055, k = 0.035):
#
#    #trade_data = trade_data.loc[trade_data.Size > 4*s,:]
#    
#    
#    trade_data = trade_data[trade_data['Size']>4*s]
#    
#    #Signal = []
#    flag=[]
#    flag_kill=[]
#    
#    #Signal_del = []
#    
#    temp_trade = []
#    temp_index = 0
#    #for i in range(trade_data.shape[0]):
#    for i in range(len(trade_data)):
#        
#    #for i in range(20):
#        #date = trade_data.index[i]
#        time = trade_data.index[i]
#        
#        book_min = book_data.loc[date,'BestAsk_price'].min()
#        book_max = book_data.loc[date,'BestBid_price'].max()
#        book_mid = (book_min + book_max)/2
#        
#        
#        temp_price = trade_data.iloc[i,:]['Price']
#        if temp_price > book_mid: #market maker sell at best ask price, pos CF
#            Signal.append(-1)
#            temp_trade.append(-1*s)
#        else:
#            Signal.append(1)
#            temp_trade.append(1*s)
#                    
#        temp = temp_position+temp_trade[i]
#        if (temp>=-k)&(temp<=j)&(temp_price >= book_max) & (temp_price <= book_min):
#            temp_position += temp_trade[i]
#            Signal_del.append(1)
#        else:
#            Signal_del.append(0)
#
#    trade_data['Trade'] = temp_trade
#    trade_data = trade_data.loc[np.array(Signal_del) == 1,:]
#    
#    trade_data['Cash'] = -1*np.cumsum(trade_data.Price*trade_data.Trade)
#    '''
#    trade_data['PnL'][0] = 0
#    trade_data['PnL'][i]= trade_data['Cash'][i-1] - trade_data['Price'][i]*trade_data['Trade'][i]
#    '''
#    
#    trade_data['Position'] = np.cumsum(trade_data.Trade)
#    trade_data['PnL'] = trade_data.Cash+trade_data.Position*trade_data.Price
#
#    '''
#    trade_data['Position'] = np.cumsum(trade_data.Trade)
#    trade_data.loc[trade_data.Position == 0,'PnL'] = trade_data.loc[trade_data.Position == 0,'Cash']
#    if trade_data.Position[0] != 0:
#        trade_data.PnL[0] = 0
#    trade_data = trade_data.fillna(method='ffill')
#    if trade_data.Position[-1] > 0:
#        trade_data.PnL[-1] = trade_data.Cash[-1]+trade_data.Position[-1]*book_data.BestAsk_price[-1]
#    if trade_data.Position[-1] < 0:
#        trade_data.PnL[-1] = trade_data.Cash[-1]+trade_data.Position[-1]*book_data.BestBid_price[-1]
#    '''
#    return trade_data




book_big = pd.read_csv('mkt_make_BTC_hw_big_2019__book_lev_2.tab',sep='\t')
book_small = pd.read_csv('mkt_make_BTC_hw_small__book_lev_2.tab',sep='\t')
                     
trade_big=pd.read_csv('mkt_make_BTC_hw_big_2019__trades.tab',sep='\t')
trade_small=pd.read_csv('mkt_make_BTC_hw_small__trades.tab',sep='\t')

trade_small = trade_small.rename(columns={'timestamp_utc_microseconds':'Time'})
trade_small1=trade_small.copy()

trade_small1['Time'] =trade_small['Time'].apply(timeStamp) 
trade_small['Time'] = list(map(lambda x: datetime.fromtimestamp(x/1000000), trade_small.Time))

trade_small = trade_small.set_index('Time')

book_small = book_small.rename(columns={'timestamp_utc_microseconds':'Time'})
book_small['Time'] = list(map(lambda x: datetime.fromtimestamp(x/1000000), book_small.Time))
book_small['received_utc_microseconds'] = list(map(lambda x: datetime.fromtimestamp(x/1000000), book_small.received_utc_microseconds))
book_small = book_small.set_index('Time')


#book_small['BestBid_price']= book_small.apply(lambda x:max(x['Bid1PriceMillionths'],x['Bid2PriceMillionths'],x['Bid3PriceMillionths']),axis=1)
bid_price = ['Bid1PriceMillionths', 'Bid2PriceMillionths', 'Bid3PriceMillionths']
book_small['BestBid_price']= book_small[bid_price].max(axis = 1)
ask_price = ['Ask1PriceMillionths', 'Ask2PriceMillionths', 'Ask3PriceMillionths']
book_small['BestAsk_price']= book_small[ask_price].min(axis = 1)
book_small['BestAsk_price'] /= 10e5
book_small['BestBid_price'] /= 10e5

def trade(trade_data, book_data, s = 0.01, j = 0.055, k = 0.035):
    trade_data = trade_data.loc[trade_data.Size > 4*s,:]
    Signal = []
    Signal_del = []
    temp_trade = []
    temp_position = 0
    for i in range(trade_data.shape[0]):
    #for i in range(20):
        date = trade_data.index[i]
        book_min = book_data.loc[date,'BestAsk_price'].min()
        book_max = book_data.loc[date,'BestBid_price'].max()
        book_mid = (book_min + book_max)/2
        temp_price = trade_data.iloc[i,:]['Price']
        if temp_price > book_mid: #market maker sell at best ask price, pos CF
            Signal.append(-1)
            temp_trade.append(-1*s)
        else:
            Signal.append(1)
            temp_trade.append(1*s)
                    
        temp = temp_position+temp_trade[i]
        if (temp>=-k)&(temp<=j)&(temp_price >= book_max) & (temp_price <= book_min):
            temp_position += temp_trade[i]
            Signal_del.append(1)
        else:
            Signal_del.append(0)

    trade_data['Trade'] = temp_trade
    trade_data = trade_data.loc[np.array(Signal_del) == 1,:]
    
    trade_data['Cash'] = -1*np.cumsum(trade_data.Price*trade_data.Trade)
    '''
    trade_data['PnL'][0] = 0
    trade_data['PnL'][i]= trade_data['Cash'][i-1] - trade_data['Price'][i]*trade_data['Trade'][i]
    '''
    
    trade_data['Position'] = np.cumsum(trade_data.Trade)
    trade_data['PnL'] = trade_data.Cash+trade_data.Position*trade_data.Price

    '''
    trade_data['Position'] = np.cumsum(trade_data.Trade)
    trade_data.loc[trade_data.Position == 0,'PnL'] = trade_data.loc[trade_data.Position == 0,'Cash']
    if trade_data.Position[0] != 0:
        trade_data.PnL[0] = 0
    trade_data = trade_data.fillna(method='ffill')
    if trade_data.Position[-1] > 0:
        trade_data.PnL[-1] = trade_data.Cash[-1]+trade_data.Position[-1]*book_data.BestAsk_price[-1]
    if trade_data.Position[-1] < 0:
        trade_data.PnL[-1] = trade_data.Cash[-1]+trade_data.Position[-1]*book_data.BestBid_price[-1]
    '''
    return trade_data

result_small = trade(trade_small, book_small)
result_small















































