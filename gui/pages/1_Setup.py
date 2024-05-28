import os
from collections import namedtuple

import streamlit as st
from create_random_forest import create_random_forest, get_random_forest_parameters
from create_sarima import create_sarima_parameters, get_sarima_parameters
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
    Reset the metrics for different models to None in the session state.

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
    Update the metrics for selected models in the session state.

    This function first resets selected model metrics to None using the
    _reset_models_metrics() private method. Then it updates the metrics for the selected
    models in the session state with the provided metrics.

    :param models_metrics: A dictionary containing the metrics for each selected model.
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


def generate_wrapper_params() -> list[str]:
    """
    Generate wrapper parameters for the model.

    This function generates wrapper parameters required for the wrapper script
    based on the selected file, days to predict, run type, and model parameters.

    :return: A list of wrapper parameters.
    :rtype: list[str]
    """
    wrapper_params = [
        st.session_state.df,
        st.session_state.days_to_predict,
        str(run_type).lower(),
        sarima_params,
        hw_params,
        hw_smoothing_params,
        rf_params,
    ]

    return wrapper_params


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
        return utils.center_text(spinner_text), spinner_text
    else:
        spinner_text = f"Calculating {', '.join(selected_models)} models"
        return utils.center_text(spinner_text), spinner_text


########################################################################################
#   PAGE STRUCTURE                                                                     #
########################################################################################


setup_container = st.container(border=True)

setup_title_container = setup_container.container()
setup_file_container = setup_container.container()
file_info_container = setup_container.container()
file_warning_placeholder = file_info_container.empty()
file_selected_placeholder = file_info_container.empty()

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

predict_btn_col = utils.center_text(PREDICT_BTN_TEXT)

add_vertical_space()

calculation_spinner_placeholder = st.container()


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


with days_to_predict_container:
    st.write("How many days should the model predict?")
    utils.slider("days_to_predict", min_val=5, max_val=50, default=30)


with model_text_container:
    st.subheader("Models")
    st.write("Choose the models that should be used.")


with advanced_container:
    with model_input_container:
        selected_models = utils.multiselect(
            "model",
            options=["Sarima", "Random Forest", "Holt-Winter"],
            default=["Sarima", "Random Forest", "Holt-Winter"],
        )

    with parameter_container:
        if selected_models:
            tabs = st.tabs(selected_models)

            for tab, model_name in zip(tabs, selected_models):
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
                "**Accurate:** Utilizes TimeSeriesSplit for a more precise \
                  evaluation of model performance across the dataset.",
                unsafe_allow_html=True,
            )

    with type_container:
        run_type = utils.radio(
            "type", options=["Forecast", "Test", "Accurate"], horizontal=True
        )
        st.session_state.selected_type = run_type.lower()

with predict_btn_col:
    predict_btn = st.button(
        PREDICT_BTN_TEXT, disabled=st.session_state.disable_btn, type="primary"
    )


########################################################################################
#   FUNCTIONALITY                                                                      #
########################################################################################

if selected_models and st.session_state.selected_file:
    st.session_state.disable_btn = False
else:
    st.session_state.disable_btn = True


if file:
    st.session_state.df = utils.read_data(file)
    st.session_state.df.to_csv(
        os.path.join("output", "latest_history.csv"), index=False
    )
    st.session_state.selected_file = file.name


if not selected_models:
    with model_warning_container:
        st.warning("Please choose a model.", icon="⚠️")
        st.session_state.disable_btn = True


if st.session_state.selected_file:
    if not file:
        with file_selected_placeholder.container():
            st.info(f"**Selected File: {st.session_state.selected_file}**")
else:
    with file_warning_placeholder.container():
        st.warning("Please select a file.", icon="⚠️")


if predict_btn:
    sarima_params, hw_params, hw_smoothing_params, rf_params = get_model_parameters(
        selected_models
    )

    wrapper_params = generate_wrapper_params()

    forecast_days = (
        st.session_state.days_to_predict
        if st.session_state.selected_type == utils.SelectedType.FORECAST
        else 0
    )
    utils.set_iframe_timestamps(forecast_days)

    spinner_col, spinner_text = set_spinner_text(selected_models)
    with spinner_col:
        with st.spinner(spinner_text):
            try:
                metrics = wrapper.call_wrapper(wrapper_params)
                st.write(metrics)
                update_model_metrics(metrics)
                st.session_state.selected_view = utils.SelectedView.FORECAST_VIEW
                st.switch_page("pages/2_Forecast.py")
            except Exception as e:
                _reset_models_metrics()
                st.warning("Something went wrong ...", icon="⚠️")
                st.write(e)

