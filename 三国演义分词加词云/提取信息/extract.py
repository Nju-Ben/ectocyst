#coding=utf-8
import jieba
import jieba.analyse
import jieba.posseg as pseg
import re

class Extractkeys:
 
    def removeEmoji(self, sentence):
        return re.sub('\[.*?\]\s+ ', '', sentence)
    
    def split2word(self, sentence):
        wordlist = sentence.split(' ')
        return wordlist

    def CutWithPartOfSpeech(self, sentence):
        sentence = self.removeEmoji(sentence)
        words =jieba.cut(sentence)
        wordlist=[]
        for word in words:
            wordlist.append(word)
        return wordlist

    def ExtractWord(self,wordlist):
        sentence = ','.join(wordlist)
        words = jieba.analyse.extract_tags(sentence,5)
        wordlist = []
        for w in words:
            wordlist.append(w)
        return wordlist

#去除停词
    def RemoveStopWord(self,wordlist):
        stopWords = {}.fromkeys([line.rstrip() for line in open('stopwords.txt')])        
        keywords = []
        for word in wordlist:
            if (word not in stopWords) and (len(word)>1):
                keywords.append(word)
        return keywords
    
    
def extract(text):
    ek = Extractkeys()
    wordlist = ek.CutWithPartOfSpeech(text)
    wordlist_cutstop = ek.RemoveStopWord(wordlist)
    return wordlist_cutstop


