import streamlit as st

import gui.st_utils as utils

utils.load_values()


def create_holt_winters():
    """
    Create the GUI for configuring Holt-Winters parameters.

    This function sets up the Streamlit container with various columns and buttons
    to allow the user to configure the parameters for the Holt-Winters model.
    It includes options for trend, seasonal, initialization method, damped trend,
    seasonal periods, smoothing level, smoothing seasonal, and smoothing trend.
    """
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Trend")
                st.button("Seasonal")
                st.button("Initialization Method")
            with input_col:
                utils.selectbox("trend", ["add", "mul"])
                utils.selectbox("seasonal", ["add", "mul"])
                utils.selectbox("initialization_method", ["heuristic", "estimated"])
        with col2:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Damped Trend")
                st.button("Seasonal Periods")
            with input_col:
                utils.bool_selectbox("damped_trend", index=1)
                utils.int_input("seasonal_periods", min_val=1, max_val=365, default=173)

        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Smoothing Level")
                st.button("Smoothing Seasonal")
            with input_col:
                utils.float_input("smoothing_level", default=0.53, max_val=0.98)
                utils.float_input("smoothing_seasonal", default=0.02, max_val=0.98)
        with col2:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Smoothing Trend")
            with input_col:
                utils.float_input("smoothing_trend", default=0.01, max_val=0.98)


def get_holt_winter_parameters() -> dict[str, str | int | float]:
    """
    Retrieve the configured Holt-Winters model parameters from session state.

    This function accesses the Streamlit session state to get the current values
    of the Holt-Winters model parameters configured by the user.

    :return: A dictionary containing the Holt-Winters model parameters.
    :rtype: dict[str, str | int | float]
    """
    return {
        "trend": st.session_state.trend,
        "damped_trend": st.session_state.damped_trend,
        "seasonal": st.session_state.seasonal,
        "seasonal_periods": st.session_state.seasonal_periods,
        "initialization_method": st.session_state.initialization_method,
    }


def get_holt_winter_smoothing_pararms() -> dict[str, float]:
    """
    Retrieve the configured Holt-Winters smoothing parameters from session state.

    This function accesses the Streamlit session state to get the current values
    of the Holt-Winters smoothing parameters configured by the user.

    :return: A dictionary containing the Holt-Winters smoothing parameters.
    :rtype: dict[str, float]
    """
    return {
        "smoothing_level": st.session_state.smoothing_level,
        "smoothing_seasonal": st.session_state.smoothing_seasonal,
        "smoothing_trend": st.session_state.smoothing_trend,
    }
