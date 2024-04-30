import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)


def prepare_data(data):
    data.set_index("date", inplace=True)
    data["day_of_year"] = data.index.dayofyear
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["year"] = data.index.year
    return data


class Rf:
    def __init__(self, data: pd.DataFrame, predict_range: int, rf_params: dict):
        self.data = prepare_data(data)
        self.predict_range = predict_range
        self.reset_params()
        self.set_params(rf_params)

        self.x = data[["day_of_year", "day_of_week", "month", "year"]]
        self.y = data["occupancy"]

    def predict(self):
        rf_model = RandomForestRegressor()
        rf_model.fit(self.x, self.y)
        latest_date = self.data.index.max()
        prediction_dates = [ latest_date + pd.DateOffset(days=i) for i in range(1 + self.predict_range)]
        future_features = pd.DataFrame(index=prediction_dates, columns=self.x.columns)
        future_features["day_of_week"] = [date.dayofweek for date in prediction_dates]
        future_features["day_of_year"] = [date.dayofyear for date in prediction_dates]
        future_features["month"] = [date.month for date in prediction_dates]
        future_features["year"] = [date.year for date in prediction_dates]
        future_predictions = rf_model.predict(future_features)
        future_features["predictions"] = future_predictions.astype(int)
        # Methode zur Vorhersage von Daten
        return future_features["predictions"]

    def put_dataset(self, dataset):
        self.data = prepare_data(data=dataset)
        self.x = self.data[["day_of_year", "day_of_week", "month", "year"]]
        self.y = self.data["occupancy"]
        # Hier könnten weitere Vorbereitungen für das Dataset erfolgen

    def set_daterange(self, daterange):
        self.daterange = daterange
        # Hier könnten weitere Vorbereitungen für die Daterange erfolgen

    def get_params(self):
        return self.rf_regressor_params

    def set_params(self, params: {}):
        if len(params) > 0:
            for key, val in params.items():
                self.rf_regressor_params[key] = val

    def reset_params(self):
        self.rf_regressor_params = {
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
