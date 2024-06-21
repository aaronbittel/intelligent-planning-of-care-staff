import streamlit as st

import gui.st_utils as utils

utils.load_values()


def create_sarima_parameters():
    """
    Create the GUI for configuring SARIMA parameters.

    This function sets up the Streamlit container with various columns and buttons
    to allow the user to configure the parameters for the SARIMA model.
    It includes options for trend autoregression order (p), seasonal autoregressive
    order (P), seasonal period (m), trend difference order (d), seasonal difference
    order (D), trend moving average order (q), and seasonal moving average order (Q).
    """
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            label_col, input_col = st.columns([1, 3])
            with label_col:
                st.button("p", help="Trend autoregression order")
                st.button("P", help="Seasonal autoregressive order")
                st.button(
                    "m", help="The number of time steps for a single seasonal period"
                )
            with input_col:
                utils.int_input("p", default=2)
                utils.int_input("P", default=1)
                utils.int_input("m", default=7, min_val=2, max_val=30)
        with col2:
            label_col, input_col = st.columns([1, 3])
            with label_col:
                st.button("d", help="Trend difference order")
                st.button("D", help="Seasonal difference order")
            with input_col:
                utils.int_input("d", max_val=2)
                utils.int_input("D", max_val=2)
        with col3:
            label_col, input_col = st.columns([1, 3])
            with label_col:
                st.button("q", help="Trend moving average order")
                st.button("Q", help="Seasonal moving average order")
            with input_col:
                utils.int_input("q")
                utils.int_input("Q", default=2)


def get_sarima_parameters() -> dict[str, int]:
    """
    Retrieve the configured SARIMA model parameters from session state.

    This function accesses the Streamlit session state to get the current values
    of the SARIMA model parameters configured by the user.

    :return: A dictionary containing the SARIMA model parameters.
    :rtype: dict[str, int]
    """
    return {
        "order": (st.session_state.p, st.session_state.d, st.session_state.q),
        "seasonal_order": (
            st.session_state.P,
            st.session_state.D,
            st.session_state.Q,
            st.session_state.m,
        ),
    }
