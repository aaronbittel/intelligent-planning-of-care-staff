import pandas as pd
import numpy as np
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
import pymannkendall as mk
from scipy import stats


DEFAULT_PARAMS = {
    "trend": "add",
    "damped_trend": False,
    "seasonal": "mul",
    "seasonal_periods": 350,
    "initialization_method": "heuristic",
}


class holtwinters:

    def __init__(
        self,
        data: pd.DataFrame,
        params: dict = None,
        smoothing_params: dict = None,
    ):
        if params is None:
            self.params = DEFAULT_PARAMS.copy()
        self.params = params
        self.data = data.copy()

        self.data.index.freq = "D"
        self.predict_range = 30
        if smoothing_params is not None:
            self.smoothing_params = smoothing_params
        else:
            self.smoothing_params = self.optimal_smoothing_params(
                self.data, self.predict_range
            )

    def predict(self):

        model_hw = ExponentialSmoothing(
            self.data["occupancy"], **self.params
        ).fit(**self.smoothing_params, optimized=False)
        prediction = model_hw.forecast(self.predict_range)

        print(self.smoothing_params)

        prediction = prediction.astype(int)
        return prediction

    def optimal_smoothing_params(self, data: pd.DataFrame, predict_range):

        seasonal = self.test_for_seasonality(data)
        trend = self.test_for_trend(data)

        smoothing_levels = np.arange(0.00, 0.98, 0.01)

        train_data = data[0 : len(data) - predict_range]
        test_data = data[len(data) - predict_range :]
        # Variablen für die Speicherung des besten Ergebnisses
        best_score = float("inf")
        best_smoothing_params = {}

        # Grid Search über die Parameterbereiche
        for level in smoothing_levels:
            if trend is True:
                smoothing_trends = np.arange(0.00, level, 0.01)
            else:
                smoothing_trends = [0.00]
            for trend in smoothing_trends:
                if seasonal is True:
                    smoothing_seasonals = np.arange(0.00, 1 - level, 0.01)
                else:
                    smoothing_seasonals = [0.00]
                for seasonal in smoothing_seasonals:
                    model = ExponentialSmoothing(
                        train_data["occupancy"], **self.params
                    ).fit(
                        smoothing_level=level,
                        smoothing_trend=trend,
                        smoothing_seasonal=seasonal,
                        optimized=False,
                    )
                    predictions = model.forecast(predict_range)

                    mape = mean_absolute_percentage_error(
                        test_data, predictions
                    )

                    if mape < best_score:
                        best_score = mape
                        best_smoothing_params = {
                            "smoothing_level": level,
                            "smoothing_trend": trend,
                            "smoothing_seasonal": seasonal,
                        }

        return best_smoothing_params

    def test_for_trend(self, data: pd.DataFrame):
        """
        Tests for trend in the data

        Returns:
            Boolean: True if there is an increasing trend, False otherwise
        """
        result = mk.original_test(data["occupancy"])
        # If there is no trend or the trend is decreasing, return False
        if result.trend == "decreasing" or result.trend == "no trend":
            return False
        # Otherwise, return True
        else:
            return True

    def test_for_seasonality(self, data: pd.DataFrame):
        """
        Tests for seasonality in the data

        Returns:
            Boolean: True if the data is seasonal, False otherwise
        """
        res = []

        data = data.copy()

        # Add month of the year as a column to the data
        data["month"] = data.index.month

        # For each unique month in the data
        for i in data.index.month.unique():
            # Append the occupancy data for that month to the res list
            res.append(data[data["month"] == i]["occupancy"].values)

        # Perform a Kruskal-Wallis H-test on the res list
        result = stats.kruskal(*res)

        # If the p-value of the test is greater than 0.05
        if result.pvalue > 0.05:
            # The data is not seasonal
            return False
        # Otherwise
        else:
            # The data is seasonal
            return True

    def print_params(self):
        out_text = ""
        for i in self.params:
            out_text = out_text + (f"{i}: {self.params[i]}\n")
        for i in self.smoothing_params:
            out_text = out_text + (f"{i}: {self.smoothing_params[i]}\n")
        return out_text


if __name__ == "__main__":

    health = pd.read_csv(
        "../../output/cut-data.csv",
        usecols=["dates", "occupancy"],
        index_col="dates",
        parse_dates=True,
    )

    unix_timestamp = int(time.time())
    output_file = (
        "../../output/holt_winter/holt-winter-%d.csv" % unix_timestamp
    )

    target_days = 30

    train_health = health[0 : len(health) - target_days]
    test_health = health[len(health) - target_days : len(health)]

    parameter = {
        "trend": "add",
        "damped_trend": False,
        "seasonal": "mul",
        "seasonal_periods": 60,
        "initialization_method": "estimated",
    }

    smoothing = {
        "smoothing_level": 0.8,
        "smoothing_trend": 0.0,
        "smoothing_seasonal": 0.02,
    }

    model = holtwinters(train_health, parameter, smoothing)
    test_predictions = model.predict()

    out = "target_days: 30\n"
    out = out + "parameters: \n"
    out = out + model.print_params()

    out = out + (
        f"Mean Absolute Error = {mean_absolute_error(test_health,test_predictions)}\n"
    )
    out = out + (
        f"Mean Squared Error = {mean_squared_error(test_health,test_predictions)}\n"
    )
    out = out + (
        f"Root Mean Squared Error = {np.sqrt(mean_squared_error(test_health,test_predictions))}\n"
    )
    out = out + (
        f"Mean Absolute Percentage Error = {mean_absolute_percentage_error(test_health,test_predictions)}\n"
    )

    out = out + "dates" + "," + "occupancy" + "\n"

    for i in range(len(test_predictions)):
        out = (
            out
            + str(
                (
                    train_health.index[len(train_health) - 1]
                    + pd.DateOffset(i + 1)
                ).date()
            )
            + ","
            + str(test_predictions.iloc[i])
            + "\n"
        )

    with open(output_file, "w") as f:
        f.write(out)
