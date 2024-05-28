import os

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

FORECAST_DAYS = 30


def load_data() -> pd.DataFrame:
    data_file_path = os.path.join("output", "cut-data.csv")
    return pd.read_csv(
        data_file_path,
        usecols=["date", "occupancy"],
        parse_dates=["date"],
        date_format=r"%Y-%m-%d",
    )


def predict(data: pd.DataFrame, params: dict[str, tuple[int]]) -> pd.DataFrame:
    train_data = data["occupancy"]

    order = params["order"]
    seasonal_order = params["seasonal_order"]

    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
    )

    model_fit = model.fit(disp=False)

    prediction = model_fit.forecast(steps=FORECAST_DAYS).rename("forecast")

    return pd.concat(
        [data.tail(30).reset_index(drop=True), prediction.reset_index(drop=True)],
        axis=1,
    )


def create_output_file_name(params: dict[str, tuple[int]]) -> str:
    return "".join(str(num) for val in params.values() for num in val)


def write_params_to_file(output_file_path: str, params: dict[str, tuple[int]]):
    with open(output_file_path, "w") as f:
        f.write(f"target_days = {FORECAST_DAYS}\n")
        f.write("sarima_params =\n")
        f.write(f"\torder = {params["order"]}\n")
        f.write(f"\tseasonal_order = {params["seasonal_order"]}\n")


if __name__ == "__main__":
    params = {"order": (2, 0, 3), "seasonal_order": (1, 0, 2, 5)}

    file_name = create_output_file_name(params)

    output_file_path = os.path.join("output", "sarima", f"sarima-{file_name}.csv")
    input_file_path = os.path.join("output", "cut-data.csv")

    occupancy_df = load_data()
    prediction_df = predict(occupancy_df, params)

    write_params_to_file(output_file_path, params)
    prediction_df.to_csv(
        output_file_path, mode="a", index=False, date_format=r"%Y-%m-%d"
    )
