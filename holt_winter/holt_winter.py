# dataframe opertations - pandas
import pandas as pd
import numpy as np
import time
# time series - statsmodels 
# holt winters 
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
# accuracy metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error

#TODO Dynamisch programmieren und 30 Tage in zukunft schreiben exportieren, parameter noch genauer anschauen


#data
health = pd.read_csv('../output/cut-data.csv',
    usecols=['dates','occupancy'], index_col='dates', parse_dates=True)

#output_data
unix_timestamp = int(time.time())
output_file = (
    '../output/holt_winter/holt-winter-%d.csv' % unix_timestamp
)

#parameters
health.index.freq = 'D'
target_days = 30
holt_params = {
    'trend': 'mul', 
    'damped_trend': False,
    'seasonal': 'mul', 
    'seasonal_periods': 60, 
    'initialization_method': 
    'estimated'}

#train and test
#multigrid search GridSearchCV
train_health = health[0:len(health)-target_days]
test_health = health[len(health)-target_days:len(health)]

fitted_model = ExponentialSmoothing(train_health['occupancy'],**holt_params).fit()
test_predictions = fitted_model.forecast(target_days)
test_predictions = test_predictions.astype(int)

#fill the csv with the prediction data
out = (f'target_days: {target_days}\n' )
out = out + "parameters: \n"
for i in holt_params: 
    out = out + (f'{i}: {holt_params[i]}\n')
out = out + (f'Mean Absolute Error = {mean_absolute_error(test_health,test_predictions)}\n')
out = out + (f'Mean Squared Error = {mean_squared_error(test_health,test_predictions)}\n')
out = out + (f'Root Mean Squared Error = {np.sqrt(mean_squared_error(test_health,test_predictions))}\n')
out = out + (f'Mean Absolute Percentage Error = {mean_absolute_percentage_error(test_health,test_predictions)}\n')

out = out + 'dates' + ',' + 'occupancy' + '\n'

for i in range(len(test_predictions)):
    out = out + str((train_health.index[len(train_health)-1] + pd.DateOffset(i+1)).date()) + ',' + str(test_predictions.iloc[i]) + '\n'

with open(output_file,'w') as f:
    f.write(out)
