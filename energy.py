#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/drive')

cp -r '/content/drive/My Drive/Kaggle_R' '/content'

import pandas as pd
import os
import numpy as np
import gc

x = list(os.listdir('/content/Kaggle_R'))

"""#READING DATA FROM ALL THE FILES"""

#Read data from all files in dfferent data frames
building_metadata = pd.read_csv("/content/drive/My Drive/Kaggle_R/building_metadata.csv")
sample_submission = pd.read_csv("/content/drive/My Drive/Kaggle_R/sample_submission.csv")
test = pd.read_csv("/content/drive/My Drive/Kaggle_R/test.csv")
train = pd.read_csv("/content/drive/My Drive/Kaggle_R/train.csv")
weather_test = pd.read_csv("/content/drive/My Drive/Kaggle_R/weather_test.csv")
weather_train = pd.read_csv("/content/drive/My Drive/Kaggle_R/weather_train.csv")

print("Length of train data",len(train))
print("Length of test data",len(test))

"""#COMBINING DATASETS(TRAIN, WHEATHER, BUILDING) FOR TEST AND TRAIN"""

#Combne train data in one dataframe
train1=pd.merge(train, building_metadata,  how='left', left_on=['building_id'], right_on = ['building_id'])
train2 = pd.merge(train1, weather_train,  how='left', left_on=['site_id','timestamp'], right_on = ['site_id','timestamp'])
gc.collect()

#Combne test data in one dataframe
test1=pd.merge(test, building_metadata,  how='left', left_on=['building_id'], right_on = ['building_id'])
test2 = pd.merge(test1, weather_test,  how='left', left_on=['site_id','timestamp'], right_on = ['site_id','timestamp'])
gc.collect()

"""## GRAPHS"""

train_g =train2
hist = train_g.hist(bins =100,figsize= (20,20))

"""#TIME SERIES GRAPH"""

train_g["timestamp"] = pd.to_datetime(train_g["timestamp"])
train_g["hour"] = np.uint8(train_g["timestamp"].dt.hour)
train_g["day"] = np.uint8(train_g["timestamp"].dt.day)
train_g["weekday_name"] = train_g["timestamp"].dt.weekday_name 
train_g["weekday"] = np.uint8(train_g["timestamp"].dt.weekday)
train_g["month"] = np.uint8(train_g["timestamp"].dt.month)

import matplotlib.pyplot as plt
import seaborn as sns
# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(10, 10)})

train_g = train_g.set_index('timestamp')

train_g['meter_reading'].plot(linewidth=0.8);

train_g['air_temperature'].plot(linewidth=0.8);

train_g['dew_temperature'].plot(linewidth=0.8);

train_g['sea_level_pressure'].plot(linewidth=0.8);

train_g['wind_direction'].plot(linewidth=0.8);

"""#CLEANING DATA
1. HANDLING kBTU VALUES FOR METER 0  IN TRAINING DATASET
Deleting rows where meter_readings is 0
"""

train2.drop(train2.loc[train2['meter_reading']==0].index, inplace=True)

"""converting meter_readings by multiplying with 0.2931"""

train2.loc[train2['meter'] == 0, 'meter_reading'] = train2['meter_reading']*0.2931
#train2

"""2. CLEANING: CHECKING COLUMNS WITH MORE NaN VALUES:
Drop data columns with more NaN values as they might be irrerelevant
"""

#Dropping columns because of large percentage values referring notebook on kaggle
gc.collect()
building_metadata_temp =pd.DataFrame({c:[sum(building_metadata[c].isna()),(sum(building_metadata[c].isna())/len(building_metadata[c]))*100]                            for c in building_metadata.columns} ,index=['Total','%'])
weather_train_temp =pd.DataFrame({c:[sum(weather_train[c].isna()),(sum(weather_train[c].isna())/len(weather_train[c]))*100]                             for c in weather_train.columns} ,index=['Total','%'])
weather_test_temp =pd.DataFrame({c:[sum(weather_test[c].isna()),(sum(weather_test[c].isna())/len(weather_test[c]))*100]                             for c in weather_test.columns} ,index=['Total','%'])

gc.collect()
print(building_metadata_temp)

gc.collect()
pd.set_option('display.max_columns',11)
print(weather_train_temp)

print(weather_test_temp)

"""Remove columns with maximum number of null values and not important features for performing SVR IN test and train dataset"""

# Remove columns with maximum number of null values
train2 = train2.drop(['year_built','floor_count','cloud_coverage','precip_depth_1_hr','primary_use','timestamp','site_id','building_id','meter'],axis =1)

test2 = test2.drop(['row_id','year_built','floor_count','cloud_coverage','precip_depth_1_hr','primary_use','timestamp','site_id','building_id','meter'],axis =1)

"""REDUCED TRAIN AND TEST FILES AFTER DROPPING COLUMNS"""

test2.head()

train2.head()

"""3. HANDLING NAn VALUES"""

train2.isnull().sum()

test2.isnull().sum()

"""A) FINDING INTERCEPTS FOR COLUMN air_temperature AND dew_temperature COLUMNS IN TRAIN  AND TEST DATASET:
First we are trying to find coef and intercept to find relationship btw variable and then based on the parameters imputing the values
"""

from sklearn.linear_model import LinearRegression

model_train=LinearRegression()
model_test=LinearRegression()
model_train
model_test

ind_train=train2[(train2['air_temperature'].notna()==True) & (train2['dew_temperature'].notna()==True)].index
ind_test=test2[(test2['air_temperature'].notna()==True) & (test2['dew_temperature'].notna()==True)].index

model_train.fit(pd.DataFrame({'cte':np.ones(len(ind_train)),'air':train2.loc[ind_train]['air_temperature']}),           train2.loc[ind_train]['dew_temperature'])
model_test.fit(pd.DataFrame({'cte':np.ones(len(ind_test)),'air':test2.loc[ind_test]['air_temperature']}),             test2.loc[ind_test]['dew_temperature'])

intertr=model_train.intercept_   # the intercept of the linear regression for the training datas .
coeftr=model_train.coef_         # the coefficient of the linear regression for the training datas

interts=model_test.intercept_   # the intercept of the linear regression for the test datas.
coefts=model_test.coef_         # the coefficient of the linear regression for the test datas

# impute dew_temperature missing values .
ind_train = train2[train2['dew_temperature'].isna()==True].index
ind_test = test2[test2['dew_temperature'].isna()==True].index

for j in ind_train :
    if pd.notna(train2.at[j,'air_temperature']):
        train2.at[j,'dew_temperature']=model_train.predict(np.array([1,train2.at[j,'air_temperature']]).reshape(1,-1))

for i in ind_test :
    if pd.notna(test2.at[i,'air_temperature']):
        test2.at[i,'dew_temperature']=model_test.predict(np.array([1,test2.at[i,'air_temperature']]).reshape(1,-1))
        
# impute air_temperature missing values .
air_tr= train2[train2['air_temperature'].isna()==True].index
air_ts= test2[test2['air_temperature'].isna()==True].index

for j in air_tr :
    if pd.notna(train2.at[j,'dew_temperature']):
        train2.at[j,'air_temperature']= (train2.at[j,'dew_temperature']- intertr)/coeftr[1]

for j in air_ts :
    if pd.notna(test2.at[j,'dew_temperature']):
        test2.at[j,'air_temperature']= (test2.at[j,'dew_temperature']- intertr)/coeftr[1]

"""DROPING REST NaN values"""

# drop the rest of missing values.
train2.dropna(inplace=True)

test2.isnull().sum()

"""FILLING NaN values with MEAN For test dataset"""

values = {'air_temperature': test2['air_temperature'].mean(), 'dew_temperature': test2['dew_temperature'].mean(),'sea_level_pressure':test2['sea_level_pressure'].mean(), 'wind_direction': test2['wind_direction'].mean(), 'wind_speed': test2['wind_speed'].mean()}
test2=test2.fillna(value=values)

"""#SEPERATING TRAIN DATASET INTO DATA AND LABELS"""

#Seperate data and label in X and Y for train file
X = train2.drop('meter_reading', axis=1)
y = train2['meter_reading']

X= np.array(X)
y=np.array(y)

"""#MODEL BUILDING"""



"""#KNN"""

from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor()
model.fit(X, y)

knn=model.predict(test2)
from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y,clf.predict(X))

sample_submission=sample_submission.drop(columns='meter_reading')
sample_submission['meter_reading']=knn
sample_submission.to_csv('/content/drive/My Drive/samp666.csv',index=False)

"""#DESCISION TREE"""

from sklearn import tree

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)

from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y,clf.predict(X))

sample_submission=sample_submission.drop(columns='meter_reading',axis = 1)
sample_submission['meter_reading']= clf.predict(test2)
sample_submission.to_csv('/content/drive/My Drive/samp667.csv',index=False)

"""##RANDOM FOREST"""

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

rrr=rf_model.predict(X)
from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y,rrr)

rrr=rf_model.predict(test2)

sample_submission=sample_submission.drop(columns='meter_reading',axis = 1)
sample_submission['meter_reading']= rrr
sample_submission.to_csv('/content/drive/My Drive/samp667.csv',index=False)

"""##LINEAR REGRESSION"""

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X, y)

reg.score(X, y)

m=reg.predict(X)

from sklearn.metrics import accuracy_score
accuracy_score(m.round(), y.round(), normalize=False)

reg.score(y, m)

"""## LIGHT GBM"""

import lightgbm

gbm = lgb.LGBMRegressor(num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=10)

gbm.fit(X,y)

#error Checking
from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y,gbm.predict(X))

"""##ADABOOST"""

from sklearn.ensemble import AdaBoostRegressor
regr = AdaBoostRegressor(random_state=0, n_estimators=100)

regr.fit(X, np.array(y,))

from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y,regr.predict(X))

"""##GRADIENTBOOSTING"""

from sklearn.ensemble import GradientBoostingRegressor

est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01, max_depth=1, random_state=0, loss='ls').fit(X,y)

from sklearn.metrics import mean_squared_log_error
mean_squared_log_error(y,est.predict(X))

