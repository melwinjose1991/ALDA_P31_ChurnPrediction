# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 23:16:42 2017

@author: Govardhan
"""

import datetime
import numpy as np
import pandas as pd
from collections import Counter
import sys
import nltk
#from sklearn.preprocessing import LabelEncoder
#from catboost import CatBoostClassifier
# --------------------- Initializing variables

fdist = nltk.FreqDist(your_list) # creates a frequency distribution from a list
most_common = fdist.max()    # returns a single element
top_three = fdist.keys()[:3] # returns a list


sys.modules[__name__].__dict__.clear()
start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading datasets

train = pd.read_csv("..\\data\\train.csv", header=0)
train_msno = list(set(train['msno']))

test = pd.read_csv("..\\data\\sample_submission_zero.csv", header=0)
test_msno = list(set(test['msno']))

unqSet = train_msno.extend(test_msno)

data = Counter(train['msno'])
data.most_common(10)

dup = pd.read_csv("..\\data\\dummy.csv", header=0)
dup['day'] = pd.to_datetime(dup['date'], format='%Y%m%d', errors='ignore').dt.day
dup['weekday'] = pd.to_datetime(dup['date'], format='%Y%m%d', errors='ignore').dt.weekday
dup_msno = list(set(dup['msno']))

t=dup.groupby(['msno', 'weekday']).sum()

list1 = train[train['msno'].isin(dup_msno)]

del train
del test
del train_msno
del test_msno
del third    


##################################################

#Aggregating data for one user

##################################################
train = pd.read_csv("..\\data\\UserLogs\\allUsersLogsGrouped.csv", header=0)
train_msno = list(set(train['msno']))

missingList = []
df = pd.DataFrame()

reqMsno = train_msno[55000:]
for id in set(reqMsno):
    userData=train[train['msno']==id]
    
    if len(userData)!=7:
        missingList.append(id)
        continue
    num_25_df = userData.pivot(index='msno', columns='weekday', values='num_25')
    num_25_df.columns = ['sun_25', 'mon_25', 'tue_25', 'wed_25', 'thu_25', 'fri_25', 'sat_25']
    
    num_50_df = userData.pivot(index='msno', columns='weekday', values='num_50')
    num_50_df.columns = ['sun_50', 'mon_50', 'tue_50', 'wed_50', 'thu_50', 'fri_50', 'sat_50']
    
    num_75_df = userData.pivot(index='msno', columns='weekday', values='num_75')
    num_75_df.columns = ['sun_75', 'mon_75', 'tue_75', 'wed_75', 'thu_75', 'fri_75', 'sat_75']
    
    num_985_df = userData.pivot(index='msno', columns='weekday', values='num_985')
    num_985_df.columns = ['sun_985', 'mon_985', 'tue_985', 'wed_985', 'thu_985', 'fri_985', 'sat_985']
    
    num_100_df = userData.pivot(index='msno', columns='weekday', values='num_100')
    num_100_df.columns = ['sun_100', 'mon_100', 'tue_100', 'wed_100', 'thu_100', 'fri_100', 'sat_100']
    
    
    num_unq_df=userData.pivot(index='msno', columns='weekday', values='num_unq')
    num_unq_df.columns = ['sun_unq', 'mon_unq', 'tue_unq', 'wed_unq', 'thu_unq', 'fri_unq', 'sat_unq']
    
    totalSec_df=userData.pivot(index='msno', columns='weekday', values='total_secs')
    totalSec_df.columns = ['sun_totalSec', 'mon_totalSec', 'tue_totalSec', 'wed_totalSec', 'thu_totalSec', 'fri_totalSec', 'sat_totalSec']
    
    m1 = num_25_df.join(num_50_df, how='outer')
    m2 = m1.join(num_75_df, how='outer')
    m3 = m2.join(num_985_df, how='outer')
    m4 = m3.join(num_100_df, how='outer')
    m5 = m4.join(num_unq_df, how='outer')
    m6 = m5.join(totalSec_df, how='outer')
    
    del m1
    del m2
    del m3
    del m4
    del m5
    del num_25_df
    del num_50_df
    del num_75_df
    del num_100_df
    del num_985_df
    del num_unq_df
    del totalSec_df
    
    df = df.append(m6)
    del m6
    gc.collect()
