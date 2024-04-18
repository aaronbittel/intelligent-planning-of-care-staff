# dataframe opertations - pandas
import pandas as pd
import numpy as np
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels 
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose 
# holt winters 
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# accuracy metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

#
health  = pd.read_csv('C:/Users/nguye/Documents/git/intelligent-planning-of-care-staff/output/descriptive_analysis.csv', usecols=['dates','occupancy'], index_col='dates', parse_dates=True)

health.index.freq = 'D'
train_health = health[50:700]
train_health
test_health = health[700:728]

fitted_model = ExponentialSmoothing(train_health['occupancy'],trend='mul',seasonal='mul',seasonal_periods=290).fit()
test_predictions = fitted_model.forecast(28)

print(f'Mean Absolute Error = {mean_absolute_error(test_health,test_predictions)}')
print(f'Mean Squared Error = {mean_squared_error(test_health,test_predictions)}')
print(f'Root Mean Squared Error = {np.sqrt(mean_squared_error(test_health,test_predictions))}')
print(f'Mean Absolute Percentage Error = {mean_absolute_percentage_error(test_health,test_predictions)}')

