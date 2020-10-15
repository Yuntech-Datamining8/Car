# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:42:53 2020

@author: HUAN
"""
#前置步驟
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import tree
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
import graphviz
import pydotplus
import openpyxl
from openpyxl import Workbook


df = pd.read_csv("car.data")
df.columns = ['buying','maint','doors','persons','lug_boot','safety','class']
df2 = df

#資料前處理onehot encoding, class屬於有序的資料因此用1234來劑型排序
pd.get_dummies(df2['buying'])
buying_encoding = pd.get_dummies(df2['buying'], prefix = 'buying')
df2 = df2.drop('buying', 1)

pd.get_dummies(df2['maint'])
maint_encoding = pd.get_dummies(df2['maint'], prefix = 'maint')
df2 = df2.drop('maint', 1)

pd.get_dummies(df2['doors'])
doors_encoding = pd.get_dummies(df2['doors'], prefix = 'doors')
df2 = df2.drop('doors', 1)

pd.get_dummies(df2['persons'])
persons_encoding = pd.get_dummies(df2['persons'], prefix = 'persons')
df2 = df2.drop('persons', 1)

pd.get_dummies(df2['lug_boot'])
lug_boot_encoding = pd.get_dummies(df2['lug_boot'], prefix = 'lug_boot')
df2 = df2.drop('lug_boot', 1)

pd.get_dummies(df2['safety'])
safety_encoding = pd.get_dummies(df2['safety'], prefix = 'safety')
df2 = df2.drop('safety', 1)

#使用1234排序
class_mapping = {'unacc':1,'acc':2,'good':3,'vgood':4}
df2['class'] =  df2['class'].map(class_mapping)


#讀檔合併onehot encoding
df2 = pd.concat([buying_encoding,maint_encoding,doors_encoding,persons_encoding,
                 lug_boot_encoding,safety_encoding,df2],axis=1)
#資料正規化
# df3 = preprocessing.normalize(df2, norm='l2')
# scaler = MinMaxScaler()
# df4 = scaler.fit(df2)
# df4 = scaler.transform(df2)

#建立特徵X，與目標y
X = df2.drop('class',axis = 1)
X_norm = preprocessing.normalize(X, norm='l2')
y = df2['class']

# #將資料區分成訓練集與測試集，可自行設定區分的百分比
X_train,X_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2)

#初步調整
clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=5)

#用建立好的模型來預測資料
df2_clf = clf.fit(X_train, y_train)

# 預測
test_y_predicted = df2_clf.predict(X_test)


# 結果
accuracy = metrics.accuracy_score(y_test, test_y_predicted)
print(accuracy)

dot_tree = tree.export_graphviz(df2_clf,out_file=None)
graph = pydotplus.graph_from_dot_data(dot_tree)
graph.write_pdf("car-gini.pdf")


df2.to_excel('car.xlsx',sheet_name='car')





