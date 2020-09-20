# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 13:50:27 2019

@author: 1
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 18:06:59 2019

@author: 1
"""
#导入需要用到的包
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#from numba import jit
#初始化参数
species = 200
iters = 5000

def getListMaxNumIndex(num_list,topk=int(0.2*species)):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    num_dict = {}
    for i in range(len(num_list)):
        num_dict[i] = num_list[i]
    res_list = sorted(num_dict.items(),key=lambda e:e[1])
    max_num_index = [one[0] for one in res_list[::-1][:topk]]
    return max_num_index

#适应度函数
def calfit(trip, num_city):
    total_dis = 0
    for i in range(num_city):
        cur_city = trip[i]
        next_city = trip[i+1] % num_city
        temp_dis = distance[cur_city][next_city]
        total_dis = total_dis + temp_dis    
    return 1 / total_dis

def dis(trip, num_city):
    total_dis = 0
    for i in range(num_city):
        cur_city = trip[i]
        next_city = trip[i+1] % num_city
        temp_dis = distance[cur_city][next_city]
        total_dis = total_dis + temp_dis   
    return total_dis
#交叉函数
def crossover(father,mother):
    num_city = len(father)
    #indexrandom = [i for i in range(int(0.4*cronum),int(0.6*cronum))]
    index_random = [i for i in range(num_city)]
    pos = random.choice(index_random)
    son1 = father[0:pos]
    son2 = mother[0:pos]
    son1.extend(mother[pos:num_city])
    son2.extend(father[pos:num_city])
    
    index_duplicate1 = []
    index_duplicate2 = []
    
    for i in range(pos, num_city):
        for j in range(pos):
            if son1[i] == son1[j]:
                index_duplicate1.append(j)
            if son2[i] == son2[j]:
                index_duplicate2.append(j)
    num_index = len(index_duplicate1)
    for i in range(num_index):
        son1[index_duplicate1[i]], son2[index_duplicate2[i]] = son2[index_duplicate2[i]], son1[index_duplicate1[i]]
    
    return son1,son2

#变异函数
def mutate(sample):
    num_city = len(sample)
    part = np.random.choice(num_city,2,replace=False)
    if part[0] > part[1]:
        max_ = part[0]
        min_ = part[1]
    else:
        max_ = part[1]
        min_ = part[0]
    after_mutate = sample[0:min_]
    temp_mutate = list(reversed(sample[min_:max_]))
    after_mutate.extend(temp_mutate)
    after_mutate.extend(sample[max_:num_city])
    return after_mutate



#读取城市位置数据
import datetime

starttime = datetime.datetime.now()

#long running


df1 = pd.read_csv('size38.txt', sep=' ', header=None)
#df1 = pd.read_csv('size29.txt', sep=' ', header=None)
#df1 = pd.read_csv('size194.txt', sep=' ', header=None)
#df1 = pd.read_csv('size131.txt', sep=' ', header=None)
#df2 = pd.read_csv('st70.txt', sep=' ', header=None)
df1[0] = df1[0] - 1
plot = plt.plot(df1[1], df1[2], '*')

#计算各城市邻接矩阵。
n = len(df1)
distance = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        temp1 = np.power((df1.iloc[i,1] - df1.iloc[j,1]),2)
        temp2 = np.power((df1.iloc[i,2] - df1.iloc[j,2]),2)
        distance[i][j] = np.sqrt(temp1 + temp2)

#初始化种群，生成可能的解的集合
x = []
counter = 0
#for i in range(species):
while counter < species:
    dna = np.random.permutation(range(n)).tolist()
    start = dna[0]
    dna.append(start)
    if dna not in x:
        x.append(dna)
        counter = counter + 1

ctlist = []
dislist = []
ct = 0
while ct < iters: 
    ct = ct + 1    
    f = []
    #p=[]
    for i in range(species):    
        f.append(calfit(x[i], n))
    
    
    #计算选择概率
    sig = sum(f)
    p = (f / sig).tolist()

    #for i in range(species):
    #   p.append(f[i]/sig)
    
    #轮盘赌选择
    
    test = getListMaxNumIndex(p)
    testnum = len(test)
    newx = []
    for i in range(testnum):
        newx.append(x[test[i]])
        #newx.append(x[test[i]])
    index = [i for i in range(species)]
    news = random.choices(index,weights=p,k=int(0.8*species))
    newsnum = len(news)
    for i in range(newsnum):
        newx.append(x[news[i]])
    
    #排序筛选
#    test=getListMaxNumIndex(p)
#    newx=[]
#    for i in range(int(len(test)/3)):
#        newx.append(x[test[i]])
#        newx.append(x[test[i]])
#        newx.append(x[test[i]])
#    for i in range(int(len(test)/3),len(test)):
#        newx.append(x[test[i]])
    
    #index=[i for i in range(species)]
    #news=random.choices(index,weights=p,k=int(0.9*species))
    #for i in range(len(news)):
        #newx.append(x[news[i]])
        
    
    #进行交叉
    #for i in range(species):
        #j=species-i-1
        
    m = int(species/2)
    for i in range(0,m):
        j = i + m - 1
        #j=i+1
        numx = len(newx[0])
        if random.choice([1,2,3,4,5,6,7,8,9,10]) < 8:
            tplist1 = newx[i][0:numx-1]
            tplist2 = newx[j][0:numx-1]
            crosslist1,crosslist2 = crossover(tplist1,tplist2)
#        else:
#            tplist1=newx[i][0:numx-1]
#            tplist2=newx[j][0:numx-1]
#            crosslist1=tplist1
#            crosslist2=tplist2
            if random.choice([1,2,3,4,5,6,7,8,9,10]) < 4:
                crosslist1 = mutate(crosslist1)
                crosslist2 = mutate(crosslist2)
            end1 = crosslist1[0]
            end2 = crosslist2[0]
            crosslist1.append(end1)
            crosslist2.append(end2)
            newx[i] = crosslist1
            newx[j] = crosslist2
    x = newx
    res = []
    for i in range(species):    
        res.append(calfit(x[i], n))  
    result = 1 / max(res)
    res1 = []
    for i in range(species):    
        res1.append(dis(x[i], n))  
    result1 = min(res1)
    print(ct)
    print(result)
    print(result1)
    ctlist.append(ct)
    dislist.append(result)
    #print(x)

#res=[]
#for i in range(species):    
#        res.append(calfit(x[i],n))
#result=1/max(f)
endtime = datetime.datetime.now()

print (endtime - starttime)

plk1 = []
plk2 = []
for i in range(len(x[0])):
    plk2.append(df1.iloc[x[0][i], 2])
    plk1.append(df1.iloc[x[0][i], 1])
plot = plt.plot(plk1, plk2, c='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
    
plot = plt.plot(ctlist, dislist)
plt.xlabel('iters')
plt.ylabel('distance')
plt.show()
    
    
