# -*- coding: utf-8 -*-
"""
@author: SACHIN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

memberdata = pd.read_csv('members.csv')
traindata = pd.read_csv('train.csv')

def transformType(dataFrame):
    list_features=list(dataFrame.select_dtypes(include=[np.number]).columns)
    
    for feature in list_features:
        currfeature=dataFrame[feature]
        maxValFeature=np.max(dataFrame[feature])
        minValFeature=np.min(dataFrame[feature])
        
        if(maxValFeature<=127  and minValFeature >= -128):
         dataFrame[feature]=currfeature.astype(np.int8)
        elif(maxValFeature<=32767  and minValFeature >= -32768):
         dataFrame[feature]=currfeature.astype(np.int16)            
        elif(maxValFeature<=2147483647  and minValFeature >= -2147483648):
         dataFrame[feature]=currfeature.astype(np.int32)
        else:    
         dataFrame[feature]=currfeature.astype(np.int64)


#converting datatype to reduce memory consumption	 
transformType(memberdata) 

#merging members data file with training file on msno parameter
trainingData=pd.merge(left=traindata,right=memberdata,how='left',on='msno')   

#deleting variable not in use to free some memory
del memberdata
del traindata

#impute missing values to city
trainingData['city']=trainingData['city'].apply(lambda x:int(x) if pd.notnull(x) else -1)
#impute missing values and treat outlier
trainingData['bd']=trainingData['bd'].apply(lambda x: int(x) if pd.notnull(x) and 0<x <100  else -1)
trainingData['registered_via']=trainingData['registered_via'].apply(lambda x: int(x) if pd.notnull(x) else -1)

#converting datatype to reduce memory consumption
trainingData['is_churn']=trainingData['is_churn'].astype(np.int8);  
trainingData['city']=trainingData['city'].astype(np.int8);  
trainingData['bd']=trainingData['bd'].astype(np.int8);  

#converting gender to numerical variable
gender = {'male':1, 'female':2}
trainingData['gender'] = trainingData['gender'].map(gender)

trainingData = trainingData.fillna(0)

trainingData['gender']=trainingData['gender'].astype(np.int8);

#generating correlation matrix
corrmatrix=trainingData[trainingData.columns[1:]].corr()
f,ax=plt.subplots(figsize=(12,9))
sns.heatmap(corrmatrix, vmax=1, cbar=True, annot=True, square=True);
plt.show()

#checking gender data relationship with churn
gender_crosstab=pd.crosstab(trainingData['gender'],trainingData['is_churn'])
gender_crosstab.plot(kind='bar', stacked=True, grid=True)

#checking age data relationship with churn
age_crosstab=pd.crosstab(trainingData['bd'],trainingData['is_churn'])
age_crosstab.plot(kind='bar', stacked=True, grid=True)

#checking city data relationship with churn
city_crosstab=pd.crosstab(trainingData['city'],trainingData['is_churn'])
city_crosstab.plot(kind='bar', stacked=True, grid=True)

#checking city data relationship with churn
registered_via_crosstab=pd.crosstab(trainingData['registered_via'],trainingData['is_churn'])
registered_via_crosstab.plot(kind='bar', stacked=True, grid=True)

trainingData.to_csv('MembersProc_file.csv', sep=',')