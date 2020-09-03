# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import auc
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('logistics.csv',engine='python')

data['result'].replace(['高','等',"低"] , [0,0,1] , inplace=True)

columns=data.columns.tolist()
x_columns=columns[0:13]

X_train,X_test,y_train,y_test = train_test_split(data[x_columns],data['result'],test_size=0.3,random_state=0)

Lr = LogisticRegression(C=1000,random_state=0)
#Lr = LogisticRegression(C=1000,random_state=0,multi_class='multinomial',solver='newton-cg')

ml= Lr.fit(X_train,y_train)
y_score=ml.predict_proba(X_test)  
y_test=np.array(y_test)

params=ml.coef_
print('特征相关系数: ')
print(params)

#ans = Lr.predict(X_test)

fpr, tpr, thresholds=sklearn.metrics.roc_curve(y_test,y_score[:,1],pos_label=1)
print('\nROC曲线面积AUC:')
AUC = auc(fpr, tpr)
print(AUC)

print('\nROC曲线')
plt.plot(fpr,tpr,marker = 'o')

plt.show()


#print(classification_report(ans, y_test))