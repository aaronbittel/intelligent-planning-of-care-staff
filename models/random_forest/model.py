import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

# Variables
# In/Out:
occupancy_source = "../../output/cut-data.csv"
unix_timestamp = int(time.time())
output_file = "../../output/random-forest/random-forest-%d.csv" % unix_timestamp
# relevant for model
target_days = 31
rf_regressor_params = {
    "n_estimators": 250,
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
    "monotonic_cst": None,
}

# Load CSV, set date as index
data = pd.read_csv(occupancy_source)
data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
data.set_index("date", inplace=True)
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

rf_model = RandomForestRegressor(**rf_regressor_params)
rf_model.fit(x_train, y_train)
latest_date = data.index.max()
prediction_dates = [
    latest_date + pd.DateOffset(days=i) for i in range(1 + target_days * -1, 1)
]
future_features = pd.DataFrame(index=prediction_dates, columns=x.columns)
future_features["day_of_year"] = [date.dayofyear for date in prediction_dates]
future_features["day_of_week"] = [date.dayofweek for date in prediction_dates]
future_features["month"] = [date.month for date in prediction_dates]
future_features["year"] = [date.year for date in prediction_dates]
future_predictions = rf_model.predict(future_features)
future_features["predictions"] = future_predictions.astype(int)

out = "target_days = %s\nrf_regressor_params = \n" % target_days
for k, v in rf_regressor_params.items():
    out += f"\t{k} = {v}\n"
out += "date,occupancy,prediction\n"
for index in y_test.index:
    out += "%s,%i,%i\n" % (
        index,
        y_test.loc[index],
        future_features["predictions"].loc[index],
    )
    print(
        "%s > expected=%i, predicted=%i"
        % (index, y_test.loc[index], future_features["predictions"].loc[index])
    )

if out.endswith("\n"):
    out = out[:-1]

rmse = root_mean_squared_error(y_test.tail(target_days), future_features["predictions"])
mea = mean_absolute_error(y_test.tail(target_days), future_features["predictions"])
mape = mean_absolute_percentage_error(
    y_test.tail(target_days), future_features["predictions"]
)

print("Root Mean Squared Error: ", rmse)
print("Mean Absolute Error: ", mea)
print("Mean Absolute Percentage Error: ", mape)

print(out)
with open(output_file, "w") as f:
    f.write(out)
