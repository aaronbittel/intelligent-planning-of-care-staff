import streamlit as st
import gui.st_utils as utils

utils.load_values()


def create_holt_winters():
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Trend")
                st.button("Seasonal")
            with input_col:
                utils.selectbox("trend", ["add", "mul"])
                utils.selectbox("seasonal", ["add", "mul"])
        with col2:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Damped Trend")
                st.button("Seasonal Periods")
            with input_col:
                utils.bool_selectbox("damped_trend", index=1)
                utils.int_input("seasonal_periods", min_val=1, max_val=365, default=60)
        col3, col4, _, _ = st.columns([3, 2.5, 2, 2])
        with col3:
            st.button("Initialization Method")
        with col4:
            utils.selectbox("initialization_method", ["heuristic", "estimated"])
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Smoothing Level")
                st.button("Smoothing Seasonal")
            with input_col:
                utils.float_input("smoothing_level", default=0.89, max_val=0.98)
                utils.float_input("smoothing_seasonal", max_val=0.98)
        with col2:
            label_col, input_col = st.columns(2)
            with label_col:
                st.button("Smoothing Trend")
            with input_col:
                utils.float_input("smoothing_trend", max_val=0.98)


def get_holt_winters_parameters():
    return {
        "trend": st.session_state.trend,
        "damped_trend": st.session_state.damped_trend,
        "seasonal": st.session_state.seasonal,
        "seasonal_periods": st.session_state.seasonal_periods,
        "initialization_method": st.session_state.initialization_method,
    }


def get_holt_winters_smoothing_pararms():
    return {
        "smoothing_level": st.session_state.smoothing_level,
        "smoothing_seasonal": st.session_state.smoothing_seasonal,
        "smoothing_trend": st.session_state.smoothing_trend,
    }