import wrapper
import os
import pandas as pd

# Test wrapper script with default parameters
if __name__ == "__main__":
    # Relative path to dataset
    input_output_folder = "../output"
    input_file = "landkreise/01001.csv"
    output_file = "latest_history.csv"
    current_directory = os.getcwd()
    test_call_wrapper_dir = os.path.dirname(os.path.abspath(__file__))
    call_relpath = os.path.relpath(test_call_wrapper_dir, current_directory)
    input_path = os.path.join(call_relpath, input_output_folder, input_file)
    output_path = os.path.join(call_relpath, input_output_folder, output_file)
    df = pd.read_csv(input_path, usecols=["date", "occupancy"], parse_dates=["date"])
    df.to_csv(output_path, index=False)

    prediction_days = 30
    type = "forecast"

    sarima_params = {
        "order": (2, 0, 0),
        "seasonal_order": (2, 0, 1, 7),
    }

    rf_params = {
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

    wh_params = {
        "trend": "add",
        "damped_trend": False,
        "seasonal": "mul",
        "seasonal_periods": 60,
        "initialization_method": "heuristic",
    }

    wh_smoothing_params = {
        "smoothing_level": 0.89,
        "smoothing_trend": 0.0,
        "smoothing_seasonal": 0.0,
    }

    # Build the command - advanced command
    command = [
        df,
        prediction_days,
        type,
        sarima_params,  # sarima_params, optional
        wh_params,  # optional,
        wh_smoothing_params,  # optional (only in combination with wh_params)
        rf_params,  # optional
    ]

    wrapper.call_wrapper(command)
