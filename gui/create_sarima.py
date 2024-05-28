import streamlit as st
import gui.st_utils as utils

utils.load_values()


def create_sarima_parameters():
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
                utils.int_input("m", default=7)
        with col2:
            label_col, input_col = st.columns([1, 3])
            with label_col:
                st.button("d", help="Trend difference order")
                st.button("D", help="Seasonal difference order")
            with input_col:
                utils.int_input("d")
                utils.int_input("D")
        with col3:
            label_col, input_col = st.columns([1, 3])
            with label_col:
                st.button("q", help="Trend moving average order")
                st.button("Q", help="Seasonal moving average order")
            with input_col:
                utils.int_input("q")
                utils.int_input("Q", default=2)


def get_sarima_parameters():
    return {
        "order": (st.session_state.p, st.session_state.d, st.session_state.q),
        "seasonal_order": (
            st.session_state.P,
            st.session_state.D,
            st.session_state.Q,
            st.session_state.m,
        ),
    }
