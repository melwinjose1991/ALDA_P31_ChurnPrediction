# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 11:10:41 2017

@author: Govardhan
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.cross_validation import train_test_split

from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score

df = pd.DataFrame()

trainDup = pd.read_csv("..\\data\\UserLogs\\allUsersLogsGrouped.csv", header=0)
trainDup_msno = list(set(trainDup['msno']))

num_25_df = trainDup.pivot(index='msno', columns='weekday', values='num_25')
num_25_df.columns = ['sun_25', 'mon_25', 'tue_25', 'wed_25', 'thu_25', 'fri_25', 'sat_25']

num_50_df = trainDup.pivot(index='msno', columns='weekday', values='num_50')
num_50_df.columns = ['sun_50', 'mon_50', 'tue_50', 'wed_50', 'thu_50', 'fri_50', 'sat_50']

num_75_df = trainDup.pivot(index='msno', columns='weekday', values='num_75')
num_75_df.columns = ['sun_75', 'mon_75', 'tue_75', 'wed_75', 'thu_75', 'fri_75', 'sat_75']

num_985_df = trainDup.pivot(index='msno', columns='weekday', values='num_985')
num_985_df.columns = ['sun_985', 'mon_985', 'tue_985', 'wed_985', 'thu_985', 'fri_985', 'sat_985']

num_100_df = trainDup.pivot(index='msno', columns='weekday', values='num_100')
num_100_df.columns = ['sun_100', 'mon_100', 'tue_100', 'wed_100', 'thu_100', 'fri_100', 'sat_100']

num_unq_df=trainDup.pivot(index='msno', columns='weekday', values='num_unq')
num_unq_df.columns = ['sun_unq', 'mon_unq', 'tue_unq', 'wed_unq', 'thu_unq', 'fri_unq', 'sat_unq']

totalSec_df=trainDup.pivot(index='msno', columns='weekday', values='total_secs')
totalSec_df.columns = ['sun_totalSec', 'mon_totalSec', 'tue_totalSec', 'wed_totalSec', 'thu_totalSec', 'fri_totalSec', 'sat_totalSec']

m1 = num_25_df.join(num_50_df, how='outer')
del num_25_df
del num_50_df
gc.collect()

m2 = m1.join(num_75_df, how='outer')
del m1
del num_75_df
gc.collect()

m3 = m2.join(num_985_df, how='outer')
del m2
del num_985_df
gc.collect()

m4 = m3.join(num_100_df, how='outer')
del m3
del num_100_df
gc.collect()

m5 = m4.join(num_unq_df, how='outer')
del m4
del num_unq_df
gc.collect()

m6 = m5.join(totalSec_df, how='outer')
del m5
del totalSec_df
gc.collect()

m6.to_csv("..\\data\\UserLogs\\allAggregateUsers.csv")