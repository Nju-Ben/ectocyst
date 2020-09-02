#coding=utf-8
__author__ = 'root'
#from PMI import *
import os
from extract import extract
import pandas as pd
from collections import Counter


def cal_wordFreq(wordlist):
    word_dict = Counter(wordlist)
    return word_dict

if __name__ == '__main__':
    documents = []
    #testfile = 'data.txt'
    testfile = '三国.txt'
    f = open(testfile, 'r',encoding='UTF-8')
    data = f.readlines()
    if data is not None:
        for sentences in data:
            extractwords = []
            words = extract(sentences)
            for word in words:
                extractwords.append(word)
            if len(extractwords) > 1:
                documents.append(extractwords)
    
    total_word = sum(documents,[])
    word_frequency = cal_wordFreq(total_word)
    word_dataframe = pd.DataFrame.from_dict(word_frequency,orient = 'index').reset_index()
    word_dataframe = word_dataframe.rename(columns={0:'frequency'})
    word_dataframe = word_dataframe.rename(columns={'index':'name'})
    word_dataframe.sort_values(by="frequency",ascending=False,inplace=True)
    word_dataframe.to_csv('word.csv',encoding = 'gbk')



            
