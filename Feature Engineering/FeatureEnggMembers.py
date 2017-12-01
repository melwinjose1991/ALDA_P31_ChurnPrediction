"""
@author: SACHIN
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

trainingData = pd.read_csv('MembersProc_file.csv')
   
#new features creation
trainingData['registration_init_year']=trainingData['registration_init_time'].apply(lambda x:int(str(x)[:4]) if pd.notnull(x) else -1)
trainingData['registration_init_month']=trainingData['registration_init_time'].apply(lambda x:int(str(x)[4:6]) if pd.notnull(x) else -1)
trainingData['registration_init_day']=trainingData['registration_init_time'].apply(lambda x:int(str(x)[6:8]) if pd.notnull(x) else -1)

trainingData['registration_init_year']=trainingData['registration_init_year'].astype(np.int16);  
trainingData['registration_init_month']=trainingData['registration_init_month'].astype(np.int8);
trainingData['registration_init_day']=trainingData['registration_init_day'].astype(np.int8);    

trainingData['expiration_year']=trainingData['expiration_date'].apply(lambda x:int(str(x)[:4]) if pd.notnull(x) else -1)
trainingData['expiration_month']=trainingData['expiration_date'].apply(lambda x:int(str(x)[4:6]) if pd.notnull(x) else -1)
trainingData['expiration_day']=trainingData['expiration_date'].apply(lambda x:int(str(x)[6:8]) if pd.notnull(x) else -1)

trainingData['expiration_year']=trainingData['expiration_year'].astype(np.int16);  
trainingData['expiration_month']=trainingData['expiration_month'].astype(np.int8);
trainingData['expiration_day']=trainingData['expiration_day'].astype(np.int8);   

#dropping old features not in use as new features are created out of it  
trainingData = trainingData.drop('registration_init_time', 1)
trainingData = trainingData.drop('expiration_date', 1)

#filling zero values
trainingData = trainingData.fillna(0)


trainingData.to_csv('Members_final_file.csv', sep=',')