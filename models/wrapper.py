import os
import sys
import json

from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import models.random_forest.rf as rf
import models.sarima.sarima as s
import models.holt_winter.holt_winter as hw
import pandas as pd
import numpy as np

# Gather input
# Determine path to wrapper script directory
# Absolute path from wrapper script
wrapper_dir = os.path.dirname(os.path.abspath(__file__))
# relative path from wrapper script to output folder
output_folder_path = os.path.join(wrapper_dir, "..", "output")
prediction_days_default = 30
type_default = "forecast"
# relative path from wrapper script to input folder
input_folder_path = os.path.join(wrapper_dir, "..", "output")
# relative path from input folder to input file
input_file_path = ""
prediction_days = 0
type = ""
advanced = False
sarima_params = {}
wh_params = {}
wh_smoothing_params = {}
rf_params = {}
rf_default_params = {
    "n_estimators": 1,
    "criterion": "squared_error",
    "max_depth": 1,
    "min_samples_split": 2,
    "min_samples_leaf": 5,
    "min_weight_fraction_leaf": 0.0,
    "max_features": "log2",
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

sarima_default_params = {
    "order": (2, 0, 0),
    "seasonal_order": (1, 0, 2, 7),
}

# Build models and conduct predictions
def build_models_and_predict(
    train_data,
    prediction_days,
    wh_params,
    wh_smoothing_params,
    rf_params,
    sarima_params,
):
    # Test whether advanced parameters have been set or not
    if not advanced:
        rf_model = rf.Rf(train_data.copy(deep=True), prediction_days, {})
        hw_model = hw.holtwinters(train_data, prediction_days)
        sarima_model = s.Sarima(train_data, prediction_days)

        prediction_rf = "Random-Forest", rf_model.predict()
        prediction_hw = "Holt-Winter", hw_model.predict()
        prediction_sarima = "Sarima", sarima_model.predict()

    # If True use the advanced parameters to fit and calculate the output.
    else:
        rf_model = (
            rf.Rf(train_data.copy(deep=True), prediction_days, {})
            if rf_params
            else None
        )
        hw_model = (
            hw.holtwinters(
                train_data,
                prediction_days,
                wh_params,
                smoothing_params=wh_smoothing_params,
            )
            if wh_params
            else None
        )
        sarima_model = (
            s.Sarima(train_data, prediction_days, sarima_params)
            if sarima_params
            else None
        )

        prediction_rf = "Random-Forest", rf_model.predict() if rf_model else None
        prediction_hw = "Holt-Winter", hw_model.predict() if hw_model else None
        prediction_sarima = "Sarima", sarima_model.predict() if sarima_model else None

    return (
        rf_model,
        hw_model,
        sarima_model,
        prediction_rf,
        prediction_hw,
        prediction_sarima,
    )


# Calculate Error metrics if selected:
def calculate_metrics(test_data, *predictions):
    metrics = {}
    for prediction in predictions:
        if isinstance(
            prediction, tuple
        ):  # Überprüfe, ob es sich um ein Tupel handelt (Name, Vorhersage)
            model_name, prediction_data = prediction

        if prediction_data is not None:
            metrics[model_name] = {
                "RMSE": root_mean_squared_error(
                    test_data["occupancy"], prediction_data["occupancy"]
                ),
                "MAPE": mean_absolute_percentage_error(
                    test_data["occupancy"], prediction_data["occupancy"]
                ),
                "MAE": mean_absolute_error(
                    test_data["occupancy"], prediction_data["occupancy"]
                ),
            }

    return metrics


# Execute basic test if selected
def setup_test(setup_test_data):
    split_day = setup_test_data["date"].max() - pd.DateOffset(days=prediction_days)
    test_data = setup_test_data[setup_test_data["date"] > split_day]
    train_data = setup_test_data[setup_test_data["date"] <= split_day]
    return test_data, train_data


# Execute advanced test (timeseries_split) if selected
# Predictiondays über geben und als test size
def setup_and_calculate_accurate(setup_accurate_data, prediction_days):
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3, test_size=prediction_days)

    rmse_per_model = {}
    mape_per_model = {}
    mae_per_model = {}

    average_rmse_per_model = {}
    average_mape_per_model = {}
    average_mae_per_model = {}
    formatted_metrics = {}

    # Split data using Time Series Split
    for train_index, test_index in tscv.split(setup_accurate_data):
        train_data, test_data = (
            setup_accurate_data.iloc[train_index],
            setup_accurate_data.iloc[test_index],
        )

        # Ensure modification doesn't affect the original data
        train_data = train_data.copy()

        # Ensure modification doesn't affect the original data
        test_data = test_data.copy()
        (
            rf_model,
            hw_model,
            sarima_model,
            prediction_rf,
            prediction_hw,
            prediction_sarima,
        ) = build_models_and_predict(
            train_data,
            prediction_days,
            wh_params,
            wh_smoothing_params,
            rf_params,
            sarima_params,
        )
        metrics = calculate_metrics(
            test_data, prediction_rf, prediction_hw, prediction_sarima
        )

        # Iterate over models in metrics
        for model_name, metrics_data in metrics.items():
            # Extract RMSE- and MAPE-values for current model
            rmse_value = metrics_data["RMSE"]
            mape_value = metrics_data["MAPE"]
            mae_value = metrics_data["MAE"]

            # Add RMSE- and MAPE-values for current model to the corresponding list
            rmse_per_model.setdefault(model_name, []).append(rmse_value)
            mape_per_model.setdefault(model_name, []).append(mape_value)
            mae_per_model.setdefault(model_name, []).append(mae_value)

    # Iterate over the models in rmse_per_model and mape_per_model
    for model_name in rmse_per_model.keys():
        # Calculate the average of the RMSE and MAPE values for the current model
        average_rmse = np.mean(rmse_per_model[model_name])
        average_mape = np.mean(mape_per_model[model_name])
        average_mae = np.mean(mae_per_model[model_name])

        # Save the average RMSE and MAPE value per model in the corresponding dictionaries
        average_rmse_per_model[model_name] = average_rmse
        average_mape_per_model[model_name] = average_mape
        average_mae_per_model[model_name] = average_mae

    for model_name in average_rmse_per_model:
        formatted_metrics[model_name] = {
            "RMSE": average_rmse_per_model[model_name],
            "MAPE": average_mape_per_model[model_name],
            "MAE" : average_mae_per_model[model_name],
        }

    return (
        rf_model,
        hw_model,
        sarima_model,
        prediction_rf,
        prediction_hw,
        prediction_sarima,
        formatted_metrics,
    )


# Remove old predictions
def remove_files_if_exist(*file_paths):
    for file_path in file_paths:
        if os.path.exists(file_path):
            os.remove(file_path)


# Write output
def write_output(prediction_sarima=None, prediction_hw=None, prediction_rf=None):
    remove_files_if_exist(
        os.path.join(output_folder_path, "latest_random_forest.csv"),
        os.path.join(output_folder_path, "latest_holt_winter.csv"),
        os.path.join(output_folder_path, "latest_sarima.csv"),
    )

    if (
        prediction_rf is not None
        and prediction_rf[1] is not None
        and not prediction_rf[1].empty
    ):
        prediction_rf[1].to_csv(
            os.path.join(output_folder_path, "latest_random_forest.csv"), index=False
        )

    if (
        prediction_hw is not None
        and prediction_hw[1] is not None
        and not prediction_hw[1].empty
    ):
        prediction_hw[1].to_csv(
            os.path.join(output_folder_path, "latest_holt_winter.csv"), index=False
        )

    if (
        prediction_sarima is not None
        and prediction_sarima[1] is not None
        and not prediction_sarima[1].empty
    ):
        prediction_sarima[1].to_csv(
            os.path.join(output_folder_path, "latest_sarima.csv"), index=False
        )
    print("Success: New model output generated")


"""
    List of parameters:
    csv_file (required),
    target days (required),
    type(required),
    sarima_params (optional),
    wh_params(optional),
    wh_smoothing_params(optional),
    rf_params(optional)
"""


def call_wrapper(params):
    global output_folder_path
    global input_folder_path
    global input_file_path
    global prediction_days
    global type
    global advanced
    global sarima_params
    global wh_params
    global wh_smoothing_params
    global rf_params

    match len(params):
        case 1:
            # No default parameters and forecast
            advanced = False
            df = params[0]
            prediction_days = prediction_days_default
            type = type_default
            sarima_params = None
            wh_params = None
            wh_smoothing_params = None
            rf_params = None

        # No advanced parameters are transferred
        case 3:
            advanced = False
            df = params[0]
            prediction_days = params[1]
            type = params[2]

        # Advanced parameters are transferred
        case 7:
            df = params[0]
            prediction_days = params[1]
            type = params[2]
            advanced = True
            sarima_params = params[3]
            wh_params = params[4]
            wh_smoothing_params = params[5]
            rf_params = params[6]
        case _:
            raise ValueError("Invalid parameter length")

    # Create relative path from input_folder to input_file
    # input_file_path = os.path.join(input_folder_path, input_file)

    # Test which type of output is to be generated
    match type:
        case "forecast":
            (
                rf_model,
                hw_model,
                sarima_model,
                prediction_rf,
                prediction_hw,
                prediction_sarima,
            ) = build_models_and_predict(
                df,
                prediction_days,
                wh_params,
                wh_smoothing_params,
                rf_params,
                sarima_params,
            )
            write_output(
                prediction_sarima=prediction_sarima,
                prediction_hw=prediction_hw,
                prediction_rf=prediction_rf,
            )
        
            test_data, train_data = setup_test(df)
            (
                rf_model,
                hw_model,
                sarima_model,
                prediction_rf,
                prediction_hw,
                prediction_sarima,
            ) = build_models_and_predict(
                train_data,
                prediction_days,
                wh_params,
                wh_smoothing_params,
                rf_params,
                sarima_params,
            )
            metrics = calculate_metrics(
                test_data, prediction_rf, prediction_hw, prediction_sarima
            )
            print(metrics)
            return metrics

        case "test":
            test_data, train_data = setup_test(df)
            (
                rf_model,
                hw_model,
                sarima_model,
                prediction_rf,
                prediction_hw,
                prediction_sarima,
            ) = build_models_and_predict(
                train_data,
                prediction_days,
                wh_params,
                wh_smoothing_params,
                rf_params,
                sarima_params,
            )
            metrics = calculate_metrics(
                test_data, prediction_rf, prediction_hw, prediction_sarima
            )
            write_output(
                prediction_sarima=prediction_sarima,
                prediction_hw=prediction_hw,
                prediction_rf=prediction_rf,
            )
            print(metrics)
            return metrics
        
        case "accurate":
            (
                rf_model,
                hw_model,
                sarima_model,
                prediction_rf,
                prediction_hw,
                prediction_sarima,
                formatted_metrics,
            ) = setup_and_calculate_accurate(df, prediction_days)
            print(formatted_metrics)
            write_output(
                prediction_sarima=prediction_sarima,
                prediction_hw=prediction_hw,
                prediction_rf=prediction_rf,
            )
            return formatted_metrics

        case _:
            raise ValueError("Invalid type")

if __name__ == "__main__":
    call_wrapper(sys.argv)
