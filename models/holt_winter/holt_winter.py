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
    "seasonal_periods": 60,
    "initialization_method": "heuristic",
}

DEFAULT_SMOOTHING_PARAMS = {
    "smoothing_level": 0.89, 
    "smoothing_trend": 0.0, 
    "smoothing_seasonal": 0.0
}

class holtwinters:

    def __init__(
        self,
        data: pd.DataFrame,
        predict_range: int = 30,
        params: dict = None,
        smoothing_params: dict = None,
    ):
        self.params = params
        if params is None:
            self.params = DEFAULT_PARAMS.copy()

        self.data = data.copy()
        self.data.set_index("date", inplace=True)
        self.data.index.freq = "D"
        self.predict_range = predict_range
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
        ret = pd.DataFrame({"date": prediction.index, "occupancy": prediction.astype(int)})
        prediction = prediction.astype(int)
        return ret

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
