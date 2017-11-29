# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:40:17 2017

@author: Govardhan
"""

import datetime
import numpy as np
import pandas as pd
from collections import Counter
import gc

#from sklearn.preprocessing import LabelEncoder
#from catboost import CatBoostClassifier
# --------------------- Initializing variables

import sys
sys.modules[__name__].__dict__.clear()
start_time = datetime.datetime.now()
print("Started at: " + str(start_time))


# --------------------- Loading datasets
train = pd.read_csv("..\\data\\train.csv", header=0)
train_msno = list(set(train['msno']))

test = pd.read_csv("..\\data\\sample_submission_zero.csv", header=0)
test_msno = list(set(test['msno']))

#Combining train and test users
reqUsers = list(set(train_msno) | set(test_msno))

#delete unnecessary data
del train
del test
del train_msno
del test_msno
gc.collect()
 
#Read first log file
logsTrain = pd.read_csv("..\\data\\UserLogs\\user_logs-000.csv", header=0)

#Find required users rows
logsTrainSmall = logsTrain[logsTrain['msno'].isin(reqUsers)]

#Write required users data to file.
logsTrainSmall.to_csv("..\\data\\UserLogs\\logs_small0.csv", index = False)

del logsTrain
del logsTrainSmall
gc.collect()



#Read second log file
logsTrain = pd.read_csv("..\\data\\UserLogs\\user_logs-001.csv", header=0)

#Find required users rows
logsTrainSmall = logsTrain[logsTrain['msno'].isin(reqUsers)]

#Write required users data to file.
logsTrainSmall.to_csv("..\\data\\UserLogs\\logs_small1.csv", index = False)

del logsTrain
del logsTrainSmall
gc.collect()


#Read third log file
logsTrain = pd.read_csv("..\\data\\UserLogs\\user_logs-002.csv", header=0)

#Find required users rows
logsTrainSmall = logsTrain[logsTrain['msno'].isin(reqUsers)]

#Write required users ..\\data to file.
logsTrainSmall.to_csv("..\\data\\UserLogs\\logs_small2.csv", index = False)

del logsTrain
del logsTrainSmall
gc.collect()


#Read fourth log file
logsTrain = pd.read_csv("..\\data\\UserLogs\\user_logs-003.csv", header=0)

#Find required users rows
logsTrainSmall = logsTrain[logsTrain['msno'].isin(reqUsers)]

#Write required users data to file.
logsTrainSmall.to_csv("..\\data\\UserLogs\\logs_small3.csv", index = False)

del logsTrain
del logsTrainSmall
gc.collect()


#Read fifth log file
logsTrain = pd.read_csv("..\\data\\UserLogs\\user_logs-004.csv", header=0)

#Find required users rows
logsTrainSmall = logsTrain[logsTrain['msno'].isin(reqUsers)]

#Write required users ..\\data to file.
logsTrainSmall.to_csv("..\\data\\UserLogs\\logs_small4.csv", index = False)

del logsTrain
del logsTrainSmall
gc.collect()


#Read sixth log file
logsTrain = pd.read_csv("..\\data\\UserLogs\\user_logs-005.csv", header=0)

#Find required users rows
logsTrainSmall = logsTrain[logsTrain['msno'].isin(reqUsers)]

#Write required users data to file.
logsTrainSmall.to_csv("..\\data\\UserLogs\\logs_small5.csv", index = False)

del logsTrain
del logsTrainSmall
gc.collect()



#Read first file data
logsTrain = pd.read_csv("..\\data\\UserLogs\\logs_small0.csv", header=0)
logsTrain['weekday'] = pd.to_datetime(logsTrain['date'], format='%Y%m%d', errors='ignore').dt.weekday

#Group by for each user
allUsersLogsG0=logsTrain.groupby(['msno', 'weekday']).sum()

del logsTrain
gc.collect()

#Read Second file data
logsTrain = pd.read_csv("..\\data\\UserLogs\\logs_small1.csv", header=0)
logsTrain['weekday'] = pd.to_datetime(logsTrain['date'], format='%Y%m%d', errors='ignore').dt.weekday

#Group by for each user
allUsersLogsG1=logsTrain.groupby(['msno', 'weekday']).sum()

del logsTrain
gc.collect()

allUsersLogs01 = allUsersLogsG0.append(allUsersLogsG1)
del allUsersLogsG0
del allUsersLogsG1

allUsersLogsG01=allUsersLogs01.groupby(['msno', 'weekday']).sum()
del allUsersLogs01
gc.collect()
########First and second files combined

#Read third file ..\\data
logsTrain = pd.read_csv("..\\data\\UserLogs\\logs_small2.csv", header=0)
logsTrain['weekday'] = pd.to_datetime(logsTrain['date'], format='%Y%m%d', errors='ignore').dt.weekday

#Group by for each user
allUsersLogsG2=logsTrain.groupby(['msno', 'weekday']).sum()

del logsTrain
gc.collect()

#Read fourth file data
logsTrain = pd.read_csv("..\\data\\UserLogs\\logs_small3.csv", header=0)
logsTrain['weekday'] = pd.to_datetime(logsTrain['date'], format='%Y%m%d', errors='ignore').dt.weekday

#Group by for each user
allUsersLogsG3=logsTrain.groupby(['msno', 'weekday']).sum()

del logsTrain
gc.collect()

allUsersLogs23 = allUsersLogsG2.append(allUsersLogsG3)
del allUsersLogsG2
del allUsersLogsG3

allUsersLogsG23=allUsersLogs23.groupby(['msno', 'weekday']).sum()
del allUsersLogs23

#Combine 0,1 AND 2,3
allUsersLogs0123=allUsersLogsG01.append(allUsersLogsG23)

del allUsersLogsG01
del allUsersLogsG23

allUsersLogsG0123 = allUsersLogs0123.groupby(['msno', 'weekday']).sum()

del allUsersLogs0123

###### 0,1,2,3 are grouped

allUsersLogsG0123.to_csv("..\\data\\UserLogs\\allUsersLogsG0123.csv", index = False)


#Read fifth file data
logsTrain = pd.read_csv("..\\data\\UserLogs\\logs_small4.csv", header=0)
logsTrain['weekday'] = pd.to_datetime(logsTrain['date'], format='%Y%m%d', errors='ignore').dt.weekday

#Group by for each user
allUsersLogsG4=logsTrain.groupby(['msno', 'weekday']).sum()

del logsTrain
gc.collect()

#Read sixth file data
logsTrain = pd.read_csv("..\\data\\UserLogs\\logs_small5.csv", header=0)
logsTrain['weekday'] = pd.to_datetime(logsTrain['date'], format='%Y%m%d', errors='ignore').dt.weekday

#Group by for each user
allUsersLogsG5=logsTrain.groupby(['msno', 'weekday']).sum()

del logsTrain
gc.collect()

allUsersLogs45 = allUsersLogsG4.append(allUsersLogsG5)
del allUsersLogsG4
del allUsersLogsG5
gc.collect()

allUsersLogsG45 = allUsersLogs45.groupby(['msno', 'weekday']).sum()
del allUsersLogs45
gc.collect()

allUsersLogs = allUsersLogsG0123.append(allUsersLogsG45)
del allUsersLogsG0123
del allUsersLogsG45

allUsersLogsG = allUsersLogs.groupby(['msno', 'weekday']).sum()

del allUsersLogs
#

allUsersLogsG.to_csv("data\\UserLogs\\allUsersLogsGrouped.csv")



#Check if we have all users data
#Read train users
train = pd.read_csv("..\\data\\train.csv", header=0)
train_msno = list(set(train['msno']))

#Read collected users
logs = pd.read_csv("..\\data\\UserLogs\\allUsersLogsGrouped.csv", header=0)
logs_msno = list(set(train['msno']))

#check if train_msno-logs_msno is null?
len(list(set(train_msno) & set(logs_msno)))
#All users are present

test = pd.read_csv("..\\data\\sample_submission_zero.csv", header=0)
test_msno = list(set(test['msno']))

#Few users are missing (# of user don't have logs = 970960-881701)

# END
###################################################
