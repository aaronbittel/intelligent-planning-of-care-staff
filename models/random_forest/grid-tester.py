import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
import numpy as np
import itertools
import time


def iterate_parameter_combinations(parameters):
    """
    Iterate over every combination of parameters specified in the dictionary.

    Args:
        parameters (dict): A dictionary where each key maps to a list of values.

    Yields:
        dict: A dictionary representing a combination of parameter values.
    """
    # Extract keys and corresponding value lists from the parameters dictionary
    keys = list(parameters.keys())
    value_lists = [parameters[key] for key in keys]

    # Generate all combinations of parameter values using itertools.product()
    for combination in itertools.product(*value_lists):
        # Create a dictionary mapping each key to its corresponding value in the combination
        combination_dict = {keys[i]: combination[i] for i in range(len(keys))}
        yield combination_dict


start = time.time()
# Variables

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=1, stop=2000, num=16)]
# Number of features to consider at every split
max_features = ['log2', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(1, 100, num = 2)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 5, 7]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 3, 5, 7]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
grid = {'n_estimators': n_estimators,
               'max_features': max_features,
              # 'max_depth': max_depth,
              # 'min_samples_split': min_samples_split,
              # 'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
# In/Out:
occupancy_source = "../../output/cut-data.csv"
# relevant for model
target_days = 40

# Load CSV, set date as index
data = pd.read_csv(occupancy_source)
data["dates"] = pd.to_datetime(data["dates"], format="%Y-%m-%d")
data.set_index("dates", inplace=True)
data["target"] = data["occupancy"].astype(int)
# Add Columns used as features
data["day_of_year"] = data.index.dayofyear
data["day_of_week"] = data.index.dayofweek
data["month"] = data.index.month
data["year"] = data.index.year

# split data into features and target
x = data[["day_of_year", "day_of_week", "month", "year"]]
y = data["target"]
# split into training and test data
x_train = x.iloc[: x.shape[0] - target_days]
x_test = x.iloc[x.shape[0] - target_days :]
y_train = y.iloc[: x.shape[0] - target_days]
y_test = y.iloc[x.shape[0] - target_days :]


latest_date = data.index.max()
prediction_dates = [
    latest_date + pd.DateOffset(days=i) for i in range(1 + target_days * -1, 1)
]
future_features = pd.DataFrame(index=prediction_dates, columns=x.columns)
future_features["day_of_year"] = [date.dayofyear for date in prediction_dates]
future_features["day_of_week"] = [date.dayofweek for date in prediction_dates]
future_features["month"] = [date.month for date in prediction_dates]
future_features["year"] = [date.year for date in prediction_dates]

good_runs = []
count = 0
best_rmse = 20
runs = 1

for param in grid:
    runs *= len(grid[param])

for params in iterate_parameter_combinations(grid):
    rf_model = RandomForestRegressor(**params)
    rf_model.fit(x_train, y_train)
    future_predictions = rf_model.predict(future_features)
    rmse = root_mean_squared_error( y_test.tail(target_days), future_predictions )
    if rmse < best_rmse:
        best_rmse = rmse
        print(rmse, params)
        good_runs.append(params)
    count+=1
    print(count, '/', runs)

latest_date = data.index.max()
prediction_dates = [
    latest_date + pd.DateOffset(days=i) for i in range(1 + target_days * -1, 1)
]

print(best_rmse)
print(good_runs[-1])
print(time.time() - start)