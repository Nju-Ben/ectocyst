
#%matplotlib inline
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

#book_big=pd.read_csv('mkt_make_BTC_hw_big__book_lev_2.tab',sep='\t')

book_big = pd.read_csv('mkt_make_BTC_hw_big_2019__book_lev_2.tab',sep='\t')
book_small = pd.read_csv('mkt_make_BTC_hw_small__book_lev_2.tab',sep='\t')
                     
                     
#trade_big=pd.read_csv('mkt_make_BTC_hw_big__trades.tab',sep='\t')
trade_big=pd.read_csv('mkt_make_BTC_hw_big_2019__trades.tab',sep='\t')
                      
trade_small=pd.read_csv('mkt_make_BTC_hw_small__trades.tab',sep='\t')

def trade(trade_data, book_data, s = 0.01, j = 0.055, k = 0.035):

    position = 0
    flag = []
    flag_del = []
    temp_trade = []
    trade_data = trade_data[trade_data['Size']>4*s]
    
    list_=book_data.index.tolist()
    
    list_index=[]
    for i in range(len(trade_data)):
        if trade_data.index[i] in list_:
            list_index.append(1)
            ask_price = book_data.loc[trade_data.index[i],'BestAsk_price'].min()
            bid_price = book_data.loc[trade_data.index[i],'BestBid_price'].max()
            trade_price = trade_data.iloc[i,0]
            price_mid = (ask_price+bid_price)/2
            if trade_price >= price_mid:
                flag.append(-1)
                temp_trade.append(-1*s)
            else:
                flag.append(1)
                temp_trade.append(1*s)

            temp = position+temp_trade[i]

            if (trade_price >= bid_price) and (trade_price <= ask_price) and (temp>=-k) and (temp<=j):
                flag_del.append(1)
                position = position+temp_trade[i]
            else:
                flag_del.append(0)
        else:
            list_index.append(0)
            
    trade_data = trade_data[np.array(list_index) == 1]
    trade_data['Trade'] = temp_trade
    trade_data = trade_data[np.array(flag_del) == 1]
    trade_data['Cash'] = -1*np.cumsum(trade_data['Trade']*trade_data['Price'])
    trade_data['Position'] = np.cumsum(trade_data['Trade'])
    trade_data['PnL'] = trade_data['Cash']+trade_data['Position']*trade_data['Price']
    
    return trade_data


def gettimeStamp(timeNum): 
    timestamp = float(timeNum/1000000)
    otherStyleTime=datetime.fromtimestamp(timestamp)
    return otherStyleTime

def processtimeStamp(timeNum):
    timeNum=int(timeNum/1000)*1000
    return timeNum

trade_small = trade_small.rename(columns={'timestamp_utc_microseconds':'time_stamp'})
trade_small['time'] =trade_small['time_stamp'].apply(gettimeStamp)
trade_small = trade_small.set_index('time')
trade_small.drop('time_stamp',axis=1, inplace=True)

trade_big = trade_big.rename(columns={'timestamp_utc_microseconds':'time_stamp'})
trade_big['time_stamp']=trade_big['time_stamp'].apply(processtimeStamp)
    
trade_big['time'] =trade_big['time_stamp'].apply(gettimeStamp)
trade_big = trade_big.set_index('time')
trade_big = trade_big.rename(columns={'size':'Size'})
trade_big = trade_big.rename(columns={'price':'Price'})
trade_big.drop('time_stamp',axis=1, inplace=True)


book_small = book_small.rename(columns={'timestamp_utc_microseconds':'time_stamp'})
book_small['time'] =book_small['time_stamp'].apply(gettimeStamp)
book_small['received_time']=book_small['received_utc_microseconds'].apply(gettimeStamp)
book_small = book_small.set_index('time')
book_small.drop('received_utc_microseconds',axis=1, inplace=True)
book_small.drop('time_stamp',axis=1, inplace=True)

book_small['BestBid_price']= book_small.apply(lambda x:max(x['Bid1PriceMillionths'],x['Bid2PriceMillionths'],x['Bid3PriceMillionths']),axis=1)
book_small['BestAsk_price']= book_small.apply(lambda x:min(x['Ask1PriceMillionths'],x['Ask2PriceMillionths'],x['Ask3PriceMillionths']),axis=1)

book_small['BestAsk_price'] = book_small['BestAsk_price']/(1000000)
book_small['BestBid_price'] =book_small['BestBid_price'] /(1000000)

book_big = book_big.rename(columns={'timestamp_utc_microseconds':'Time'})
epoch = datetime(1970, 1, 1)
book_big['Time'] = list(map(lambda x: datetime.fromtimestamp(x/1000000), book_big.Time))
book_big['received_utc_microseconds'] = list(map(lambda x: datetime.fromtimestamp(x/1000000), book_big.received_utc_microseconds))
book_big = book_big.set_index('Time')


bid_price = ['Bid1PriceMillionths', 'Bid2PriceMillionths', 'Bid3PriceMillionths']
ask_price = ['Ask1PriceMillionths', 'Ask2PriceMillionths', 'Ask3PriceMillionths']
book_big['BestBid_price']= book_big[bid_price].max(axis = 1)
book_big['BestAsk_price']= book_big[ask_price].min(axis = 1)
book_big['BestAsk_price'] /= 10e5
book_big['BestBid_price'] /= 10e5

result_small = trade(trade_small, book_small)
print(result_small)
