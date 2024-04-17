import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error


target_days = 31
occupancy_source = '../../output/cut-data.csv'
output_file = '../../output/latest-random-forest.csv'

# Load CSV, set date as index
data = pd.read_csv(occupancy_source)
data['dates'] = pd.to_datetime(data['dates'], format='%Y-%m-%d')
data.set_index('dates', inplace=True)
data['target'] = data['occupancy'].astype(int)
data['day_of_year'] = data.index.dayofyear
data['day_of_week'] = data.index.dayofweek
data['month'] = data.index.month
data['year'] = data.index.year

x = data[['day_of_year', 'day_of_week', 'month', 'year']]
y = data['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=True, random_state=42)
params = {
    "n_estimators": 100,
    "criterion": "squared_error",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "min_weight_fraction_leaf": 0.0,
    "max_features": 1.0,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "bootstrap": True,
    "oob_score": False,
    "n_jobs": None,
    "random_state": None,
    "verbose": 0,
    "warm_start": False,
    "ccp_alpha": 0.0,
    "max_samples": None,
    "monotonic_cst": None
}
rf_model = RandomForestRegressor(**params)
rf_model.fit(x_train, y_train)
latest_date = data.index.max()
prediction_dates = [latest_date + pd.DateOffset(days=i) for i in range(1 + target_days * -1, 1)]
future_features = pd.DataFrame(index=prediction_dates, columns=x.columns)
future_features['day_of_year'] = [date.dayofyear for date in prediction_dates]
future_features['day_of_week'] = [date.dayofweek for date in prediction_dates]
future_features['month'] = [date.month for date in prediction_dates]
future_features['year'] = [date.year for date in prediction_dates]
future_predictions = rf_model.predict(future_features)
future_features['predictions'] = future_predictions.astype(int)

for i in range(target_days):
    print("> expected=%.3f, predicted=%.3f" % (data['occupancy'].iloc[i], future_features['predictions'].iloc[i]))

rmse = root_mean_squared_error(data['occupancy'].tail(target_days), future_features['predictions'])
mea = mean_absolute_error(data['occupancy'].tail(target_days), future_features['predictions'])
mape = mean_absolute_percentage_error(data['occupancy'].tail(target_days), future_features['predictions'])

print("Root Mean Squared Error: ", rmse)
print("Mean Absolute Error: ", mea)
print("Mean Absolute Percentage Error: ", mape)
