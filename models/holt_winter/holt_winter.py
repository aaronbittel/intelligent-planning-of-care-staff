import pandas as pd
import numpy as np
import time
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error,root_mean_squared_error,mean_absolute_percentage_error


DEFAULT_PARAMS = {
    'trend': 'mul', 
    'damped_trend': False,
    'seasonal': 'mul', 
    'seasonal_periods': 60, 
    'initialization_method': 
    'estimated'}


class holtwinters:

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.predict_range = 30

    def predict(self):
        train_data = self.data.iloc[: -self.predict_range]["occupancy"]

        model = ExponentialSmoothing(
            train_data,
            trend=DEFAULT_PARAMS['trend'],
            damped_trend=DEFAULT_PARAMS['damped_trend'],
            seasonal=DEFAULT_PARAMS['seasonal'],
            seasonal_periods=DEFAULT_PARAMS['seasonal_periods'],
            initialization_method=DEFAULT_PARAMS['initialization_method'],
        ).fit()

        return model.forecast(steps=self.predict_range)


