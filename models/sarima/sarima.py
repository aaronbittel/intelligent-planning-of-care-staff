import datetime

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DEFAULT_PARAMS = {
    "order": (2, 0, 0),
    "seasonal_order": (1, 0, 2, 7),
}


class Sarima:
    def __init__(
        self,
        data: pd.DataFrame,
        target_days: int = 30,
        sarima_params: dict[str, tuple[int]] = DEFAULT_PARAMS,
    ) -> None:
        self.data = data
        self.target_days = target_days
        self._sarima_params = sarima_params

    @property
    def sarima_params(self) -> dict[str, tuple[int]]:
        return self._sarima_params

    @sarima_params.setter
    def sarima_params(self, new_params: dict[str, tuple[int]]) -> None:
        if self._check_valid_param("order") and self._check_valid_param(
            "seasonal_order"
        ):
            self._sarima_params = new_params

    def predict(self) -> pd.DataFrame:
        train_data = self.data["occupancy"]
        start_date = self.data["date"].iloc[-1] + datetime.timedelta(days=1)

        prediction_dates = [
            start_date + datetime.timedelta(days=i) for i in range(self.target_days)
        ]

        order = self.sarima_params["order"]
        seasonal_order = self.sarima_params["seasonal_order"]

        model = SARIMAX(
            train_data,
            order=order,
            seasonal_order=seasonal_order,
        )

        model_fit = model.fit(disp=False)

        prediction = model_fit.forecast(steps=self.target_days)
        return pd.DataFrame({"date": prediction_dates, "occupancy": prediction})

    def _check_valid_param(self, param_name: str, nr_of_ints: int) -> bool:
        if (
            param_name not in self.new_params
            or not isinstance(self.new_params[param_name], tuple)
            or len(self.new_params[param_name]) != nr_of_ints
        ):
            print(
                f"The '{param_name}' key must be present and the value must be a tuple with {nr_of_ints} ints."
            )
            return False
        return True
