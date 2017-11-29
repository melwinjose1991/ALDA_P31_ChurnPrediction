# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 15:34:33 2017

@author: Govardhan
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.cross_validation import train_test_split

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

train = pd.read_csv("..\\data\\traindata_final.csv", header=0)
train_msno = list(set(train['msno']))

test =  pd.read_csv("..\\data\\testdata_final.csv", header=0)
test_msno = list(set(test['msno']))

'''
logs = pd.read_csv("data\\UserLogs\\allAggregateUsers.csv", header=0)
logs_msno = list(set(logs['msno']))

sampledTrain = train[train['msno'].isin(logs_msno)]
testLogs = test[test['msno'].isin(logs_msno)]
missingTest_msno = test[~test['msno'].isin(logs_msno)]['msno']


trainData = pd.merge(train, logs, on='msno')
testData = pd.merge(test, logs, on='msno')
'''
'''
cols = ['msno']


for x in cols:
    train[x] = trainData[x].astype('object')

'''
# --------------------- Label Encoding
'''
encode_cols = ['msno']
for col in encode_cols:
    le = LabelEncoder()
    le.fit(list(trainData[col].values))
    trainData[col] = le.transform(list(trainData[col]))
'''

#Dropping msno
train.drop(['msno'], axis = 1, inplace = True)
validation = test
validation.drop(['msno'], axis=1, inplace = True)

# --------------------- Reordering columns

cols = train.columns.tolist()
cols.insert(len(cols), cols.pop(cols.index('is_churn')))
train = train.reindex(columns=cols)
test = test.reindex(columns=cols)

# Prepare trainX trainY
trainX = train
trainY = train['is_churn']
trainX.drop(['is_churn'], axis = 1, inplace = True)
validation.drop(['is_churn'], axis = 1, inplace = True)

# ---------------------- Cat model
X_train, X_test, y_train, y_test = train_test_split(trainX, trainY, test_size = 0.2)
model = CatBoostClassifier(depth=15, iterations=10, learning_rate=0.1, eval_metric='Logloss', random_seed=1, calc_feature_importance=True, verbose=True)

cat_cols = [0, 2, 3, 8, 9, 12, 16]
model.fit(trainX
          ,trainY
          ,cat_features=cat_cols
          ,eval_set = (X_test, y_test)
          ,use_best_model = True
         )

pred = model.predict_proba(validation)
pred_train = model.predict_proba(train)

sub_train = pd.DataFrame({'msno': train_msno, 'is_churn': pred_train[:, 1]})
sub = pd.DataFrame({'msno': test['msno'], 'is_churn': pred[:, 1]})

sub.to_csv('..\\data\\pred_cat.csv', index=False, float_format="%.6f")

log_loss(test["is_churn"], pred, labels=["msno", "is_churn"])

pred_accu = model.predict(train)
pred_accuTest = model.predict(validation)