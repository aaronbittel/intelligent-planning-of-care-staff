import os
import sys

from sklearn.metrics import root_mean_squared_error

import models.random_forest.rf as rf
import models.sarima.sarima as s
import models.holt_winter.holt_winter as hw
import pandas as pd


def read_in_csv(csv_file):
    """

    :param csv_file: csv file path
    :return: pandas dataframe with only dates and occupancy
    """
    data = pd.read_csv(csv_file)
    data["date"] = pd.to_datetime(data["dates"], format="%Y-%m-%d")
    for column in data.columns:
        if column != "occupancy" and column != "date":
            data.drop(column, axis=1, inplace=True)
    data["occupancy"] = data["occupancy"].astype(int)
    return data


def write_file(data, csv_file_path):
    """

    :param data: pd.Dataframe with occupancy and dates
    :param csv_file_path: where to save the data
    """
    with open(csv_file_path, "w") as csv_file:
        csv_file.write("date,occupancy\n")
        csv_file.write(data.to_csv(header=False))


if __name__ == "__main__":
    # Gather Input
    output_folder = "output"
    input_folder = "output/landkreise"
    input_file = "11000.csv"
    prediction_days = 32
    test = False
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test = True

    data = read_in_csv(os.path.join(input_folder, input_file))
    if test:
        split_day = data["date"].max() - pd.DateOffset(days=prediction_days)
        data_compare = data[data["date"] > split_day]
        data = data[data["date"] <= split_day]

    # Build models
    rf_model = rf.Rf(data.copy(deep=True), prediction_days, {})
    hw_model = hw.holtwinters(data, prediction_days)
    sarima_model = s.Sarima(data, prediction_days)

    # Let each model make a prediction
    prediction_rf = rf_model.predict()
    prediction_hw = hw_model.predict()
    prediction_sarima = sarima_model.predict()
    # Write Output
    prediction_rf.to_csv("output/latest_random_forest.csv", index=False)
    #write_file(prediction_hw, 'output/latest_holt_winter.csv')
    prediction_sarima.to_csv("output/latest_sarima.csv", index=False)
    if os.path.islink(os.path.join(output_folder, "latest_history.csv")):
        os.remove(os.path.join(output_folder, "latest_history.csv"))
    os.symlink(os.path.relpath(os.path.join(input_folder, input_file), output_folder), os.path.join(output_folder, "latest_history.csv"))

    # Calculate Errors if in a test scenario:
    if test:
        print("RMSE Random Forest: ", root_mean_squared_error(data_compare["occupancy"], prediction_rf["occupancy"]))
        print("RMSE Sarima: ", root_mean_squared_error(data_compare["occupancy"], prediction_sarima["occupancy"]))
        print("RMSE Holt-Winter: ", root_mean_squared_error(data_compare["occupancy"], prediction_hw["occupancy"]))
