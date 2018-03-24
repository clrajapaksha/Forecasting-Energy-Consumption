# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 11:21:16 2018

@author: Chathuranga_08290
"""

# Importing the libraries
import tensorflow as tf # module for deep learning
import numpy as np # module for numerical calculations + linear algebra
import pandas as pd # module for data processing
import matplotlib.pyplot as plt # module for plotting
import datetime as dt # module for manipulating dates and times
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.externals import joblib
from scipy import stats
from sklearn.linear_model import LinearRegression

# Data loading
data_path ='D:\Datasets\drivendata\\'

metadata = pd.read_csv(data_path+'metadata.csv')
metadata_df = pd.DataFrame(metadata)
#metadata_df.head()

training_data = pd.read_csv(data_path+'train.csv')
train_df = pd.DataFrame(training_data)
#train_df.head()

submission_frequency = pd.read_csv(data_path+'submission_frequency.csv')
submission_frequency_df = pd.DataFrame(submission_frequency)
submission_frequency_df['ForecastPeriodMin'] = submission_frequency_df['ForecastPeriodNS']/(1000000000*60)
del submission_frequency_df['ForecastPeriodNS']
#submission_frequency_df.head()

holiday_data = pd.read_csv(data_path+'holidays.csv')
holiday_df = pd.DataFrame(holiday_data)
holiday_df['Date'] = pd.to_datetime(holiday_df['Date'], format='%Y-%m-%d')
#print(type(holiday_df['Date'][0]))
#holiday_df.head()

result_df = pd.merge(train_df, metadata_df, on='SiteId')
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

result_df = result_df.fillna(result_df.mean())
result_df = result_df[((result_df['Rate'] - result_df['Rate'].mean()) / result_df['Rate'].std()).abs() < 3]

#d1 = dt.date(2016, 2, 15)
#len(holiday_df.loc[(holiday_df['Date'] == d1) & (holiday_df['SiteId']==1)])

X = result_df[['Surface', 'IsDayOff', 'Weekday', 'Week']]
X.head()
y = result_df['Rate']
y[~((y-y.mean()).abs()>3*y.std())] # remove outliers
y.head()

# train data and test data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, shuffle=False)

print("Data preprocessing is done")

# delete unnessasary dataframes
del train_df, metadata_df, submission_frequency_df
del training_data, metadata, submission_frequency
del result_df
del X, y


# Instanciate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
#gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

model = LinearRegression()
model.fit(X_train, y_train)

# saving the model
model_name = './model1-GPR.joblib.pkl'
_ = joblib.dump(model, model_name, compress=9)

# Testing model
y_predicted = model.predict(X_test)

max_value = 25000 # max(y_test)

plt.figure(figsize = (9,8))
plt.scatter(y_test, y_predicted)
plt.plot([min(y_test), max_value], [min(y_test), max_value], 'r')
plt.xlim([min(y_test), max_value])
plt.ylim([min(y_test), max_value])
plt.title('Predicted vs. observed energy consumption')
plt.xlabel('Observed')
plt.ylabel('Predicted')
plt.show()

# NWRMSE metric
def power_laws_nwrmse(actual, predicted):
    """ Calcultes NWRMSE for the Power Laws Forecasting competition.
        Data comes in the form:
        col 0: site id
        col 1: timestamp
        col 2: forecast id
        col 3: consumption value
        Computes the weighted, normalized RMSE per site and then
        averages across forecasts for a final score.
    """
    def _per_forecast_wrmse(actual, predicted, weights=None):
        """ Calculates WRMSE for a single forecast period.
        """
        # limit weights to just the ones we need
        weights = weights[:actual.shape[0]]

        # NaNs in the actual should be weighted zero
        nan_mask = np.isnan(actual)
        weights[nan_mask] = 0
        actual[nan_mask] = 0

        # calculated weighted rmse
        total_error = np.sqrt((weights * ((predicted - actual) ** 2)).sum())

        # normalized by actual consumption (avoid division by zero for NaNs)
        denom = np.mean(actual)
        denom = denom if denom != 0.0 else 1e-10
        return total_error / denom

    # flatten and cast forecast ids
    forecast_ids = actual[:, 2].ravel().astype(int)

    # flatten and cast actual + predictions
    actual_float = actual[:, 3].ravel().astype(np.float64)
    predicted_float = predicted[:, 3].ravel().astype(np.float64)

    # get the unique forecasts
    unique_forecasts = np.unique(forecast_ids)
    per_forecast_errors = np.zeros_like(unique_forecasts, dtype=np.float64)

    # pre-calc all of the possible weights so we don't need to do so for each site
    # wi = (3n –2i + 1) / (2n^2)
    n_obs = 200  # longest forecast is ~192 obs
    weights = np.arange(1, n_obs + 1, dtype=np.float64)
    weights = (3 * n_obs - (2 * weights) + 1) / (2 * (n_obs ** 2))

    for i, forecast in enumerate(unique_forecasts):
        mask = (forecast_ids == forecast)
        per_forecast_errors[i] = _per_forecast_wrmse(actual_float[mask],
                                                     predicted_float[mask],
                                                     weights=weights)

    return np.mean(per_forecast_errors)

#NWRMSE = power_laws_nwrmse(y_test, y_predicted)
#print("NWRMSE for ML model: %s", NWRMSE)
print('Done')
