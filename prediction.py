import csv
import subprocess
import models.random_forest.rf as rf
import pandas as pd

def read_in_csv(csv_file):
    """

    :param csv_file: csv file path
    :return: pandas dataframe with only dates and occupancy
    """
    data = pd.read_csv(occupancy_source)
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


if __name__ == '__main__':
    occupancy_source = "output/cut-data.csv"
    data = read_in_csv(occupancy_source)
    rf_model = rf.Rf(data, 31, {'n_estimators': 1})
    prediction_rf = rf_model.predict()
    write_file(prediction_rf, 'output/rf.csv')
    print(prediction_rf)