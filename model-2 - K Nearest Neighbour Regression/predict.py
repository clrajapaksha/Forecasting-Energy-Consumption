# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 17:15:10 2018

@author: Chathuranga_08290
"""

import numpy as np # module for numerical calculations + linear algebra
import pandas as pd # module for data processing
from sklearn.externals import joblib

# Data loading
data_path ='D:\Datasets\drivendata\\'

metadata = pd.read_csv(data_path+'metadata.csv')
metadata_df = pd.DataFrame(metadata)

submit_data = pd.read_csv(data_path+'submission_format.csv')
submit_df = pd.DataFrame(submit_data)

submission_frequency = pd.read_csv(data_path+'submission_frequency.csv')
submission_frequency_df = pd.DataFrame(submission_frequency)
submission_frequency_df['ForecastPeriodMin'] = submission_frequency_df['ForecastPeriodNS']/(1000000000*60)
del submission_frequency_df['ForecastPeriodNS']

holiday_data = pd.read_csv(data_path+'holidays.csv')
holiday_df = pd.DataFrame(holiday_data)
holiday_df['Date'] = pd.to_datetime(holiday_df['Date'], format='%Y-%m-%d')

result_df = pd.merge(submit_df, metadata_df, on='SiteId')
result_df = pd.merge(result_df, submission_frequency_df, on='ForecastId')
result_df = result_df[['obs_id', 'SiteId', 'Timestamp', 'Surface', 'FridayIsDayOff', 'SaturdayIsDayOff', 'SundayIsDayOff', 'ForecastPeriodMin', 'Value']]
result_df['Timestamp'] = pd.to_datetime(result_df['Timestamp'], format='%Y-%m-%d %H:%M:%S')
result_df['IsDayOff'] = False
result_df['Weekday'] = result_df['Timestamp'].dt.dayofweek
result_df['Week'] = result_df['Timestamp'].dt.week
result_df['Date'] = result_df['Timestamp'].dt.date

result_df['IsDayOff'] = np.where(((result_df['Weekday']==6) & (result_df['SundayIsDayOff']==True)) | ((result_df['Weekday']==5) & (result_df['SaturdayIsDayOff']==True)) | ((result_df['Weekday']==4) & (result_df['FridayIsDayOff']==True)), True, False)
#result_df.head()

#result_df['IsDayOff'] = np.where(len(holiday_df.loc[(holiday_df['Date'] == result_df['Date']) & (holiday_df['SiteId']==result_df['SiteId'])]), True)
result_df['Rate'] = result_df['Value']/result_df['ForecastPeriodMin']

del result_df['FridayIsDayOff']
del result_df['SaturdayIsDayOff']
del result_df['SundayIsDayOff']
del result_df['Value']


X = result_df[['Surface', 'IsDayOff', 'Weekday', 'Week']]

model_path ='D:\Datasets\drivendata\Forecasting-Energy-Consumption\model-2 - K Nearest Neighbour Regression\\'
model = joblib.load(model_path+'model2-KNNR.joblib.pkl')

y = model.predict(X)
y_predict = pd.merge(submit_df, submission_frequency_df, on='ForecastId')
y_predict['Rate'] = y
y_predict['Value'] = y_predict['ForecastPeriodMin']*y_predict['Rate']
del y_predict['Rate']
del y_predict['ForecastPeriodMin']


y_predict.to_csv(model_path+'submission_2.csv', index=False)
print('done')