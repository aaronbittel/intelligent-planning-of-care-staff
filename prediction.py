import os
import models.random_forest.rf as rf
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


if __name__ == '__main__':
    # Gather Input
    output_folder = "output"
    input_folder = "output"
    input_file = "cut-data.csv"
    data = read_in_csv(os.path.join(input_folder, input_file))

    # Build models
    rf_model = rf.Rf(data, 31, {'n_estimators': 1})
    #hw_model =
    #sarima_model =

    # Let each model make a prediction
    prediction_rf = rf_model.predict()
    #prediction_hw = hw_model.predict()
    #prediction_sarima = sarima_model.predict()

    # Write Output
    write_file(prediction_rf, 'output/latest_random_forest.csv')
    #write_file(prediction_hw, 'output/latest_holt_winter.csv')
    #write_file(prediction_sarima, 'output/latest_sarima.csv')
    if os.path.exists(os.path.join(output_folder, "latest_random_forest.csv")):
        os.remove(os.path.join(output_folder, "latest_history.csv"))
    os.symlink(input_file, os.path.join(output_folder, 'latest_history.csv'))

