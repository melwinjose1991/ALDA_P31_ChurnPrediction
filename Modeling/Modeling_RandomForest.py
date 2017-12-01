# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:44:55 2017

@author: SACHIN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

# reading relvant files
traindata = pd.read_csv('msno_train.csv')
testdata = pd.read_csv('msno_val.csv')
memberdata = pd.read_csv('Members_final_file.csv')


# merging memberfile with train and test set 
traindata = pd.merge(left = traindata,right = memberdata ,how = 'left',on=['msno'])
testdata= pd.merge(left = testdata,right = memberdata ,how = 'left',on=['msno'])

# dropping non-relevant features
testdata= testdata.drop('expiration_year',1)
testdata= testdata.drop('expiration_month',1)
testdata= testdata.drop('expiration_day',1)
traindata=traindata.drop('Unnamed: 0',1)
testdata=testdata.drop('Unnamed: 0',1)

testdata= testdata.drop('expiration_year',1)
testdata= testdata.drop('expiration_month',1)
testdata= testdata.drop('expiration_day',1)
traindata=traindata.drop('Unnamed: 0',1)
testdata=testdata.drop('Unnamed: 0',1)

#reading transactions file
traintransdata = pd.read_csv('train_transaction.csv')

# dropping target variable data present in transactions data since its already present in our training set
traintransdata=traintransdata.drop('is_churn',1)

# merging transactions data with train and test set 
traindata = pd.merge(left = traindata,right = traintransdata ,how = 'left',on=['msno'])
testdata= pd.merge(left = testdata,right = traintransdata ,how = 'left',on=['msno'])

#deleting variable not in use to free some memory
del traintransdata

#reading userlogs file
userlogsdata = pd.read_csv('allAggregateUsers.csv')

# merging userlogs data with train and test set 
traindata = pd.merge(left = traindata,right = userlogsdata ,how = 'left',on=['msno'])
testdata= pd.merge(left = testdata,right = userlogsdata ,how = 'left',on=['msno'])

#deleting variable not in use to free some memory
del userlogsdata

# generating correlation matrix for checking highly correlated features
corrmatrix=traindata[traindata.columns[1:]].corr()
f,ax=plt.subplots(figsize=(20,15))
sns.heatmap(corrmatrix);

# dropping highly correlated features
traindata=traindata.drop('mon_unq',1)
traindata=traindata.drop('tue_unq',1)
traindata=traindata.drop('wed_unq',1)
traindata=traindata.drop('thu_unq',1)
traindata=traindata.drop('mon_100',1)
traindata=traindata.drop('tue_100',1)
traindata=traindata.drop('wed_100',1)
traindata=traindata.drop('thu_100',1)
traindata=traindata.drop('mon_985',1)
traindata=traindata.drop('tue_985',1)
traindata=traindata.drop('wed_985',1)
traindata=traindata.drop('thu_985',1)
traindata=traindata.drop('mon_75',1)
traindata=traindata.drop('tue_75',1)
traindata=traindata.drop('wed_75',1)
traindata=traindata.drop('thu_75',1)
traindata=traindata.drop('mon_25',1)
traindata=traindata.drop('tue_25',1)
traindata=traindata.drop('wed_25',1)
traindata=traindata.drop('thu_25',1)
traindata=traindata.drop('mon_50',1)
traindata=traindata.drop('tue_50',1)
traindata=traindata.drop('wed_50',1)
traindata=traindata.drop('thu_50',1)


testdata=testdata.drop('mon_unq',1)
testdata=testdata.drop('tue_unq',1)
testdata=testdata.drop('wed_unq',1)
testdata=testdata.drop('thu_unq',1)
testdata=testdata.drop('mon_100',1)
testdata=testdata.drop('tue_100',1)
testdata=testdata.drop('wed_100',1)
testdata=testdata.drop('thu_100',1)
testdata=testdata.drop('mon_985',1)
testdata=testdata.drop('tue_985',1)
testdata=testdata.drop('wed_985',1)
testdata=testdata.drop('thu_985',1)
testdata=testdata.drop('mon_75',1)
testdata=testdata.drop('tue_75',1)
testdata=testdata.drop('wed_75',1)
testdata=testdata.drop('thu_75',1)
testdata=testdata.drop('mon_25',1)
testdata=testdata.drop('tue_25',1)
testdata=testdata.drop('wed_25',1)
testdata=testdata.drop('thu_25',1)
testdata=testdata.drop('mon_50',1)
testdata=testdata.drop('tue_50',1)
testdata=testdata.drop('wed_50',1)
testdata=testdata.drop('thu_50',1)

#imputing  missing values with 0
traindata = traindata.fillna(0)
testdata = testdata.fillna(0)

#creating train and test variables to train and test the model
train_y=traindata['is_churn']
featuresdata= traindata.drop('is_churn',1)
train_x=featuresdata

test_y=testdata['is_churn']
testfeaturesdata= testdata.drop('is_churn',1)
test_x=testfeaturesdata


# Building RandomForest model
trained_model = RandomForestClassifier(n_estimators=100,n_jobs=-1)
trained_model.fit(train_x.drop('msno',axis=1), train_y)

#getting predictions for testset from RandomForest model buit and fitted
predictions = trained_model.predict(test_x.drop('msno',axis=1))

# Printing the train and test accuracy
print ("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x.drop('msno',axis=1))))
print ("Test Accuracy  :: ", accuracy_score(test_y, predictions))
from collections import OrderedDict
prediction_df = pd.DataFrame(OrderedDict([ ("msno", test_x["msno"]), ("is_churn", predictions) ]))

#Exporting predictions to csv file
prediction_df.to_csv("prediction_split.csv", index=False)

# Generating Prediction probabilities for TestSet from Random Forest model so that it can be used for logloss and Ensembling later
predictionsprob = trained_model.predict_proba(test_x.drop('msno',axis=1))
secondpred= [item[1] for item in predictionsprob]
predictionProbdf = pd.DataFrame(OrderedDict([ ("msno", test_x["msno"]), ("is_churn", secondpred)]))

#Exporting predictions probabilities to csv file
predictionProbdf.to_csv("predictionTestProb_split.csv", index=False)

# Generating Prediction probabilities for TrainSet from Random Forest model so that it can be used for logloss and Ensembling later
predictionsTrainprob = trained_model.predict_proba(train_x.drop('msno',axis=1))
predTrain=trained_model.predict(train_x.drop('msno',axis=1))
secondTrainpred= [itemt[1] for itemt in predictionsTrainprob]
predictionTrainProbdf = pd.DataFrame(OrderedDict([ ("msno", train_x["msno"]), ("is_churn", secondTrainpred)]))

#Exporting predictions probabilities to csv file
predictionTrainProbdf.to_csv("predictionTrainProb_split.csv", index=False)
scoreTest = log_loss(test_y, predictionsprob,labels=["msno","is_churn"])
print(scoreTest)

# Getting logloss from the predictionprobabilities to generate logloss performance measure for Random Forest model built
scoreTrain = log_loss(train_y, predictionsTrainprob,labels=["msno","is_churn"])
print(scoreTrain)

# Generating Feature importance data for Random Forest model built
importances=trained_model.feature_importances_
indices = np.argsort(importances)[::-1]
trainlabels=list(train_x.columns.drop('msno',1))
importanceList=np.array((importances)).tolist()
featureList={}
for i in range(len(trainlabels)):
    featureList[trainlabels[i]] = importanceList[i]

# Getting top ten features as per feature importance generated above	
vallist = featureList.values()
vallist.sort()
import operator
sorted_d = sorted(featureList.items(), key=operator.itemgetter(1),reverse=True)

# Plotting Feature importance data for Random Forest model built      
plt.figure()
plt.rcParams['figure.figsize']=17,12
plt.title("Feature importances")
plt.yticks(range(train_x.shape[1]-1),train_x.columns.drop('msno',1))
plt.barh(indices, importances[indices],
       color="b", align="center")
plt.xlim([-1, ])
plt.xlabel('Features importance score')
plt.show()    


