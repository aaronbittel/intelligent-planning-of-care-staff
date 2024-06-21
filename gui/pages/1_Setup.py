import os
from collections import namedtuple
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

import gui.st_utils as utils
import models.wrapper as wrapper
from gui.create_holt_winter import (
    create_holt_winters,
    get_holt_winter_parameters,
    get_holt_winter_smoothing_pararms,
)
from gui.create_random_forest import create_random_forest, get_random_forest_parameters
from gui.create_sarima import create_sarima_parameters, get_sarima_parameters


########################################################################################
#   SETTING VARIABLES                                                                  #
########################################################################################


utils.on_page_load()
utils.load_values()

param_creator = namedtuple("ParamCreator", ["create_params"])

tab_contents = {
    "Sarima": param_creator(create_params=create_sarima_parameters),
    "Random Forest": param_creator(create_params=create_random_forest),
    "Holt-Winter": param_creator(create_params=create_holt_winters),
}

PREDICT_BTN_TEXT = "PREDICT"


########################################################################################
#   METHODS                                                                            #
########################################################################################


def _reset_models_metrics() -> None:
    """
    Reset the metrics for all models to None in the session state.

    This function resets the metrics for SARIMA, Random Forest, and Holt-Winter models
    to None in the session state, allowing for a clean start or reset of model metrics.
    """
    st.session_state.metrics = {
        "Sarima": {"RMSE": None, "MAPE": None},
        "Random-Forest": {"RMSE": None, "MAPE": None},
        "Holt-Winter": {"RMSE": None, "MAPE": None},
    }


def update_model_metrics(models_metrics: dict[dict[str, float]]) -> None:
    """
    Update metrics for all models with the newly calculated metrics from the wrapper.

    :param models_metrics: A dictionary containing the metrics for each model.
    :type models_metrics: dict[dict[str, float]]
    """
    _reset_models_metrics()
    for model_name, metrics in models_metrics.items():
        st.session_state.metrics[model_name] = metrics


def get_model_parameters(selected_models: list[str]) -> tuple[dict, dict, dict, dict]:
    """
    Get the parameters for selected models.

    This function retrieves the parameters for the selected models, including SARIMA,
    Holt-Winter, and Random Forest.

    :param selected_models: A list of selected models.
    :type selected_models: list[str]
    :return: A tuple containing the parameters for SARIMA, Holt-Winter,
             Holt-Winter smoothing, and Random Forest models.
    :rtype: tuple[dict, dict, dict, dict]
    """
    sarima_params = get_sarima_parameters() if "Sarima" in selected_models else {}

    hw_params = get_holt_winter_parameters() if "Holt-Winter" in selected_models else {}

    hw_smoothing_params = (
        get_holt_winter_smoothing_pararms() if "Holt-Winter" in selected_models else {}
    )

    rf_params = (
        get_random_forest_parameters() if "Random Forest" in selected_models else {}
    )

    return sarima_params, hw_params, hw_smoothing_params, rf_params


def generate_wrapper_params(
    sarima_params: dict, hw_params: dict, hw_smoothing_params: dict, rf_params: dict
) -> list[pd.DataFrame | int | str | dict]:
    """
    Generate wrapper parameters for model prediction.

    This function constructs a list of parameters required by the wrapper script
    for performing predictions based on the selected data, prediction horizon,
    prediction type, and model-specific parameters.

    :param sarima_params: Parameters for SARIMA model.
    :type sarima_params: dict
    :param hw_params: Parameters for Holt-Winters model.
    :type hw_params: dict
    :param hw_smoothing_params: Smoothing parameters for Holt-Winters model.
    :type hw_smoothing_params: dict
    :param rf_params: Parameters for Random Forest model.
    :type rf_params: dict
    :return: A list containing the following parameters:
             - DataFrame: The selected data for prediction.
             - int: Number of days to predict.
             - str: Type of prediction run (forecast, test, accurate).
             - dict: Parameters for SARIMA model.
             - dict: Parameters for Holt-Winters model.
             - dict: Smoothing parameters for Holt-Winters model.
             - dict: Parameters for Random Forest model.
    :rtype: list[pd.DataFrame | int | str | dict]
    """
    return [
        st.session_state.df,
        st.session_state.days_to_predict,
        str(st.session_state.selected_type).lower(),
        sarima_params,
        hw_params,
        hw_smoothing_params,
        rf_params,
    ]


def set_predict_button_state() -> None:
    """
    Enable or disable the predict button based on file validity and model selection.

    This function updates the `is_button_disabled` attribute in the Streamlit
    session state. The button is enabled if a valid file is uploaded and models
    are selected; otherwise, it is disabled.
    """
    if st.session_state.valid_file and st.session_state.selected_models:
        st.session_state.is_button_disabled = False
    else:
        st.session_state.is_button_disabled = True


def set_spinner_text(selected_models: list[str]) -> tuple[st.columns, str]:
    """
    Generates and centers a spinner text message based on the selected models.

    This function takes a list of selected model names and returns a tuple containing
    a Streamlit column used for centering the spinner text and a string indicating that
    the calculation is in progress for the given models. If only one model is selected,
    the message is singular; otherwise, it lists all selected models in the plural form.
    The column layout adjusts based on the length of the text to better center it.

    :param selected_models: A list of selected model names.
    :type selected_models: list[str]
    :return: A tuple containing a Streamlit column for centered text placement and a
     string message indicating the models being calculated.
    :rtype: tuple
    """
    if len(selected_models) == 1:
        spinner_text = f"Calculating {selected_models[0]} model"
        return utils.center_col(spinner_text), spinner_text
    else:
        spinner_text = f"Calculating {', '.join(selected_models)} models"
        return utils.center_col(spinner_text), spinner_text


def handle_file_upload(file: StringIO) -> None:
    """
    Handle file upload and validation.

    This function handles the upload of a CSV file, validates its contents, and updates
    the Streamlit session state accordingly. It reads the uploaded file using a utility
    function, saves it to a designated output directory as "latest_history.csv", and
    updates the displayed file name. If the file does not conform to expected structure
    (i.e., lacks columns 'date' and 'occupancy'), it raises a warning message. Any
    other exceptions encountered during file reading or processing are also captured
    and result in an appropriate warning. Finally, it sets the `valid_file` flag in
    Streamlit session state to indicate whether the uploaded file was successfully
    processed and validated.

    :param file: The StringIO object containing the uploaded CSV file.
    :type file: StringIO

    :raises ValueError: If the uploaded CSV file does not contain expected columns.
    :raises Exception: For any other unanticipated errors during file handling.
    """
    try:
        st.session_state.df = utils.read_data(file)
        st.session_state.df.to_csv(
            os.path.join("output", "latest_history.csv"), index=False
        )
        utils.update_file_name(file.name)
        st.session_state.valid_file = True
    except ValueError:
        file_warning_placeholder.container().warning(
            "The csv file must have the columns 'date' and 'occupancy'", icon="⚠️"
        )
        st.session_state.valid_file = False
    except Exception:
        file_warning_placeholder.container().warning(
            "Unable to read the file.", icon="⚠️"
        )
        st.session_state.valid_file = False


def create_bar_chart(
    data: pd.Series, x_labels: list[str], y_label: str, title: str
) -> px.bar:
    """
    Helper function to create a bar chart using Plotly.

    :param data: Data for the bar chart.
    :type data: pd.Series
    :param x_labels: Labels for the x-axis.
    :type x_labels: list
    :param y_label: Label for the y-axis.
    :type y_label: str
    :param title: Title of the chart.
    :type title: str
    :return: The generated bar chart.
    :rtype: px.bar
    """
    data.index = data.index.map(lambda x: x_labels[x - 1] if isinstance(x, int) else x)

    fig = px.bar(
        data,
        y=data.values,
        x=data.index,
        color=data.index,
        color_discrete_map={label: "#FF4B4B" for label in x_labels},
        labels={"x": y_label, "y": "Occupancy"},
        orientation="v",
        title=title,
    )
    fig.update_layout(showlegend=False)

    return fig


def create_weekly_figure() -> px.bar:
    """
    Creates a weekly occupancy figure using Plotly.

    This function generates a bar chart representing the average occupancy per weekday.
    The data is extracted from the DataFrame stored in `st.session_state.df`, which is
    grouped by weekdays.

    :return: The generated bar chart showing average occupancy per weekday.
    :rtype: px.bar
    """
    df = st.session_state.df

    df["Weekday"] = df["date"].dt.dayofweek
    weekly_data = df.groupby("Weekday")["occupancy"].mean()

    weekday_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    return create_bar_chart(
        weekly_data, weekday_names, "Weekday", "Average Occupancy per Weekday"
    )


def create_monthly_figure() -> px.bar:
    """
    Creates a monthly occupancy figure using Plotly.

    This function generates a bar chart representing the average occupancy per month.
    The data is extracted from the DataFrame stored in `st.session_state.df`, which is
    grouped by months.

    :return: The generated bar chart showing average occupancy per month.
    :rtype: px.bar
    """
    df = st.session_state.df

    df["Month"] = df["date"].dt.month
    monthly_data = df.groupby("Month")["occupancy"].mean()

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    return create_bar_chart(
        monthly_data, month_names, "Month", "Average Occupancy per Month"
    )


########################################################################################
#   PAGE STRUCTURE                                                                     #
########################################################################################


setup_container = st.container(border=True)
setup_title_container = setup_container.container()
setup_file_container = setup_container.container()
file_info_container = setup_container.container()
file_warning_placeholder = file_info_container.empty()
file_selected_placeholder = file_info_container.empty()
occupancy_analysis_placeholder = setup_container.empty()
days_to_predict_container = setup_container.container()

st.divider()
add_vertical_space()

advanced_container = st.expander("Advanced", expanded=False)
model_container = advanced_container.container(border=True)
model_text_container = model_container.container()
model_input_container = model_container.container()
model_warning_container = model_container.container()
parameter_container = advanced_container.container()
type_container = advanced_container.container(border=True)

add_vertical_space()

predict_btn_col = utils.center_col(PREDICT_BTN_TEXT)

add_vertical_space()

calculation_spinner_placeholder = st.empty()


########################################################################################
#   CREATING WIDGETS                                                                   #
########################################################################################


with setup_title_container:
    st.title("Setup")


with setup_file_container:
    st.write(
        "Upload data defining past experiences regarding the occupancy of your hospital"
        " and let our AI do its magic."
    )
    file = st.file_uploader(
        label="Upload your File here",
        type=["csv"],
        label_visibility="collapsed",
    )
    if file:
        handle_file_upload(file)
    else:
        with file_selected_placeholder.container():
            name = utils.create_display_name(st.session_state.selected_file)
            st.info(f"**Selected File: {st.session_state.file_display_name}**")

with occupancy_analysis_placeholder.container():
    with st.expander("Occupancy Analysis"):
        weekly_col, monthly_col = st.columns(2)
        weekly_graph_container = weekly_col.container(border=True)
        monthly_graph_container = monthly_col.container(border=True)
        with weekly_graph_container:
            utils.write_center("Average Occupancy per Weekday", tag="h3")

            fig_weekly = create_weekly_figure()
            st.plotly_chart(fig_weekly, use_container_width=False)

        with monthly_graph_container:
            utils.write_center("Average Occupancy per Month", tag="h3")
            fig_weekly = create_monthly_figure()
            st.plotly_chart(fig_weekly, use_container_width=False)


with days_to_predict_container:
    st.write("How many days should the model predict?")
    utils.slider("days_to_predict", min_val=5, max_val=50, default=30)


with model_text_container:
    st.subheader("Models")
    st.write("Choose the models that should be used.")


with advanced_container:
    with model_input_container:
        utils.multiselect(
            "selected_models",
            options=["Sarima", "Random Forest", "Holt-Winter"],
            default=["Sarima", "Random Forest", "Holt-Winter"],
        )

    with parameter_container:
        if st.session_state.selected_models:
            tabs = st.tabs(st.session_state.selected_models)

            for tab, model_name in zip(tabs, st.session_state.selected_models):
                with tab:
                    st.subheader("Parameters")
                    st.write("Set the parameters to maximaize model fit.")
                    st.subheader(model_name)
                    tab_contents[model_name].create_params()
                    st.subheader("Type")
                    st.write(
                        "Define the type of prediction you want to make."
                        " Note, an accurate prediction requires more time."
                    )

        with st.popover("Type Explanation"):
            st.caption("**Forecast:** Predicts the upcoming days.")
            st.caption(
                "**Test:** Evaluates the model's performance against actual data."
            )
            st.caption(
                "**Accurate:** Utilizes TimeSeriesSplit for a more precise "
                "evaluation of model performance across the dataset.",
            )

    with type_container:
        run_type = utils.radio(
            "type", options=["Forecast", "Test", "Accurate"], horizontal=True
        )
        st.session_state.selected_type = run_type.lower()


########################################################################################
#   FUNCTIONALITY                                                                      #
########################################################################################


if not st.session_state.selected_models:
    with model_warning_container:
        st.warning("Please choose a model.", icon="⚠️")


set_predict_button_state()


with predict_btn_col:
    if st.button(
        PREDICT_BTN_TEXT, disabled=st.session_state.is_button_disabled, type="primary"
    ):
        sarima_params, hw_params, hw_smoothing_params, rf_params = get_model_parameters(
            st.session_state.selected_models
        )

        wrapper_params = generate_wrapper_params(
            sarima_params, hw_params, hw_smoothing_params, rf_params
        )

        forecast_days = (
            st.session_state.days_to_predict
            if st.session_state.selected_type == utils.SelectedType.FORECAST
            else 0
        )
        utils.set_iframe_timestamps(forecast_days)

        with calculation_spinner_placeholder.container():
            spinner_col, spinner_text = set_spinner_text(
                st.session_state.selected_models
            )
            with spinner_col:
                with st.spinner(spinner_text):
                    try:
                        metrics = wrapper.call_wrapper(wrapper_params)
                        update_model_metrics(metrics)
                        st.session_state.df.to_csv(
                            os.path.join("output", "latest_history.csv"), index=False
                        )
                        st.switch_page("pages/2_Forecast.py")
                    except ValueError:
                        st.warning("Invalid parameter combination", icon="⚠️")
                    except Exception:
                        st.warning("Something went wrong ...", icon="⚠️")
