"""
This module provides utility functions for a Streamlit application, including
session state management, UI element creation, and data validation.

Streamlit's core concept is that the Python script for a page is executed from top to
bottom on every widget update. Session State is used to share variables between
different pages as well as to store and persist the state of widgets.

### How to Save the State of Widgets on Page Change

1. **Initialize the Widgets with the `key` Parameter**:
   - This automatically creates a session state variable for the widget.

2. **Copy the Variable to a Session State with the Name `_{key}`**:
   - Store the widget's value in another session state variable prefixed with an
   underscore.

3. **Handle Page Reloads**:
   - On page reload, the widget gets recreated, which can overwrite the session state of
   the key. However, the `_{key}` variable retains the old value.

4. **Use `save_value()` and `load_values()` Methods**:
   - `save_value(key)`: Saves the current value of the widget to the session state with
   the name `_{key}`.
   - `load_values()`: Loads values from session state variables prefixed with an
   underscore  back to their corresponding keys.

These methods ensure that the state of the widgets is maintained across page reloads and
interactions, providing a consistent user experience.

### Convenience Functions for UI Element Creation

The following methods provide convenience functions that wrap around Streamlit's
native methods to create UI elements with default behavior:

- `int_input`: Creates an integer input widget with default behavior.
- `float_input`: Creates a float input widget with default behavior.
- `selectbox`: Creates a selectbox widget with default behavior.
- `bool_selectbox`: Creates a boolean selectbox widget with default behavior.
- `multiselect`: Creates a multiselect widget with default behavior.
- `radio`: Creates a radio button widget with default behavior.
- `slider`: Creates a slider widget with default behavior.

These methods simplify the creation of UI elements by reducing repetitive code
and providing sensible defaults.
"""

import datetime
import os
from io import StringIO

import pandas as pd
import streamlit as st


class SelectedView:
    """Constants for different views."""

    FORECAST_VIEW = "forecast_view"
    WEEKLY_VIEW = "weekly_view"


class SelectedType:
    """Constants for different types."""

    FORECAST = "forecast"
    TEST = "test"
    ACCURATE = "accurate"


def load_values() -> None:
    """
    Overwrite all widget values with the value before the reload.

    Loads values from session state variables prefixed with an underscore (widget value)
    to their corresponding keys. This ensures that widget states are restored after a
    page reload.
    """
    for key in st.session_state:
        if key.startswith("_"):
            st.session_state[key[1:]] = st.session_state[key]


def save_value(key: str) -> None:
    """
    Save the current value of a widget to the session state with a key prefixed by an
    underscore.

    :param key: The key of the widget whose current value needs to be saved.
    :type key: str
    """
    st.session_state[f"_{key}"] = st.session_state[key]


@st.cache_data
def check_correct_csv_format(filename: str) -> bool:
    """
    Check if a CSV file has the correct format.

    This function verifies that the CSV file contains the required columns
    'dates' and 'occupancy', and that the 'dates' column can be parsed as dates.

    :param filename: The name of the CSV file to check.
    :type filename: str
    :return: True if the CSV file has the correct format, False otherwise.
    :rtype: bool
    """
    try:
        pd.read_csv(
            os.path.join("output", filename),
            usecols=["date", "occupancy"],
            parse_dates=["date"],
        )
        return True
    except ValueError:
        return False


def int_input(
    label: str, min_val: int = 0, max_val: int = 10, default: int = 0
) -> st.number_input:
    """
    Create an integer input widget.

    This function creates an integer input widget with specified label, range,
    and default value.

    :param label: The label for the input widget.
    :type label: str
    :param min_val: The minimum value for the input widget.
    :type min_val: int
    :param max_val: The maximum value for the input widget.
    :type max_val: int
    :param default: The default value for the input widget.
    :type default: int
    :return: The integer input widget.
    :rtype: st.number_input
    """
    if label not in st.session_state:
        st.session_state[label] = default

    return st.number_input(
        label=label,
        min_value=min_val,
        max_value=max_val,
        step=1,
        label_visibility="collapsed",
        key=label,
        on_change=save_value,
        args=[label],
    )


def float_input(
    label: str, min_val: float = 0.0, max_val: float = 1.0, default: float = 0.0
) -> st.number_input:
    """
    Create a float input widget.

    This function creates a float input widget with specified label, range, and
    default value.

    :param label: The label for the input widget.
    :type label: str
    :param min_val: The minimum value for the input widget.
    :type min_val: float
    :param max_val: The maximum value for the input widget.
    :type max_val: float
    :param default: The default value for the input widget.
    :type default: float
    :return: The float input widget.
    :rtype: st.number_input
    """
    if label not in st.session_state:
        st.session_state[label] = default

    return st.number_input(
        label=label,
        min_value=min_val,
        max_value=max_val,
        step=0.01,
        label_visibility="collapsed",
        key=label,
        on_change=save_value,
        args=[label],
    )


def selectbox(label: str, options: list[str], index: int = 0) -> st.selectbox:
    """
    Create a selectbox widget.

    This function creates a selectbox widget with specified label, options, and
    default index.

    :param label: The label for the selectbox widget.
    :type label: str
    :param options: The options to display in the selectbox.
    :type options: list[str]
    :param index: The default selected index.
    :type index: int
    :return: The selectbox widget.
    :rtype: st.selectbox
    """
    return st.selectbox(
        label=label,
        label_visibility="collapsed",
        options=options,
        key=label,
        index=index,
        on_change=save_value,
        args=[label],
    )


def bool_selectbox(label: str, index: int = 0) -> st.selectbox:
    """
    Create a boolean selectbox widget.

    This function creates a selectbox widget with True and False options.

    :param label: The label for the selectbox widget.
    :type label: str
    :param index: The default selected index.
    :type index: int
    :return: The boolean selectbox widget.
    :rtype: st.selectbox
    """
    return selectbox(label, options=[True, False], index=index)


def multiselect(label: str, options: list[str], default: list[str]) -> st.multiselect:
    """
    Create a multiselect widget.

    This function creates a multiselect widget with specified label, options,
    and default selections.

    :param label: The label for the multiselect widget.
    :type label: str
    :param options: The options to display in the multiselect.
    :type options: list[str]
    :param default: The default selected options.
    :type default: list[str]
    :return: The multiselect widget.
    :rtype: st.multiselect
    """
    if label not in st.session_state:
        st.session_state[label] = default

    return st.multiselect(
        label=label,
        label_visibility="collapsed",
        options=options,
        key=label,
        on_change=save_value,
        args=[label],
    )


def radio(label: str, options: list[str], horizontal: bool = False) -> st.radio:
    """
    Create a radio button widget.

    This function creates a radio button widget with specified label, options,
    and layout.

    :param label: The label for the radio button widget.
    :type label: str
    :param options: The options to display in the radio button widget.
    :type options: list[str]
    :param horizontal: Whether to display the radio buttons horizontally.
    :type horizontal: bool
    :return: The radio button widget.
    :rtype: st.radio
    """
    return st.radio(
        label=label,
        label_visibility="collapsed",
        options=options,
        key=label,
        args=[label],
        on_change=save_value,
        horizontal=horizontal,
    )


def slider(label: str, min_val: int, max_val: int, default: int) -> st.slider:
    """
    Create a slider widget.

    This function creates a slider widget with specified label, range, and
    default value.

    :param label: The label for the slider widget.
    :type label: str
    :param min_val: The minimum value for the slider.
    :type min_val: int
    :param max_val: The maximum value for the slider.
    :type max_val: int
    :param default: The default value for the slider.
    :type default: int
    :return: The slider widget.
    :rtype: st.slider
    """
    set_session_state_variable(name=label, initial_value=default)

    return st.slider(
        label=label,
        min_value=min_val,
        max_value=max_val,
        step=1,
        key=label,
        on_change=save_value,
        args=[label],
        label_visibility="collapsed",
    )


def set_session_state_variable(name: str, initial_value: any = None) -> None:
    """
    Set a session state variable if it is not already set.

    This function initializes a session state variable with a given name and
    initial value if it does not already exist in the session state.

    :param name: The name of the session state variable.
    :type name: str
    :param initial_value: The initial value of the session state variable.
    :type initial_value: any
    """
    if name not in st.session_state:
        st.session_state[name] = initial_value


def set_metrics_variable() -> None:
    """
    Set the metrics session state variable if it is not already set.

    This function initializes a session state variable for metrics if it does
    not already exist in the session state.
    """
    if "metrics" not in st.session_state:
        st.session_state.metrics = {
            "Sarima": {"RMSE": None, "MAPE": None, "MAE": None},
            "Random-Forest": {"RMSE": None, "MAPE": None, "MAE": None},
            "Holt-Winters": {"RMSE": None, "MAPE": None, "MAE": None},
        }


def read_data(file: StringIO) -> pd.DataFrame:
    """
    Reads data from an uploaded CSV file and returns a DataFrame.

    This function reads a CSV file uploaded via Streamlit's file_uploader, extracting
    the 'date' and 'occupancy' columns, and parses the 'date' column as datetime
    objects.

    :param file: The uploaded CSV file to be read.
    :type file: StringIO
    :return: A DataFrame containing the 'date' and 'occupancy' columns.
    :rtype: pd.DataFrame
    """
    return pd.read_csv(
        file,
        usecols=["date", "occupancy"],
        parse_dates=["date"],
    )


def set_df_variable() -> None:
    """
    Sets the DataFrame in the session state.

    This function reads data from 'cut-data.csv' in the 'output' directory and sets
    it as the 'df' variable in the session state. The 'cut-data.csv' file is used
    as the default CSV file.
    """
    set_session_state_variable("df", read_data(os.path.join("output", "cut-data.csv")))


def set_all_session_state_variables() -> None:
    """
    Set all necessary session state variables for the application.

    This function initializes various session state variables required by the
    application.
    """
    set_df_variable()
    set_session_state_variable("selected_view", SelectedView.FORECAST_VIEW)
    set_session_state_variable("show_selected_file", False)
    set_session_state_variable("selected_file", "cut-data.csv")
    set_session_state_variable("disable_btn", False)
    set_session_state_variable("selected_type", SelectedType.FORECAST)
    set_session_state_variable("file_name", "cut-data.csv")
    set_session_state_variable("start_timestamp")
    set_session_state_variable("end_timestamp")
    set_metrics_variable()


def set_iframe_timestamps(forecast_days: int = 0) -> None:
    """
    Set the start and end timestamps for an iframe based on the selected file.

    This function retrieves the first and last dates from the selected CSV file.
    If the selected_type is 'forecast', the forecast_days parameter specifies the
    number of forecast days. The function then calculates the start and end timestamps
    and updates them in the session state.

    If the selected_type is SelectedType.FORECAST, the end timestamp is extended by the
    forecast_days to accommodate the forecasted values. If the selected_type is 'test'
    or 'accurate', - forecast_days shoule be 0 - the end timestamp is set to the last
    date in the data.

    :param forecast_days: The number of days to extend the end timestamp.
    :type forecast_days: int
    """
    dates = st.session_state.df["date"]
    st.session_state.start_timestamp = int(dates.iloc[0].timestamp() * 1000)
    st.session_state.end_timestamp = int(
        (dates.iloc[-1] + datetime.timedelta(days=forecast_days)).timestamp() * 1000
    )


def create_iframe_link(theme: str = "light") -> str:
    """
    Create a link for an iframe with specified theme and timestamps.

    This function generates a link for an iframe that displays data within a
    specified time range and theme.

    :param theme: The theme for the iframe (e.g., "light", "dark").
    :type theme: str
    :return: The generated iframe link.
    :rtype: str
    """
    return (
        f"http://localhost:3000/d/adj16nde8uwhsa/mixed-data?orgId=1&"
        f"from={st.session_state.start_timestamp}&to={st.session_state.end_timestamp}&"
        f"theme={theme}&viewPanel=1"
    )


def on_page_load() -> None:
    """
    Set the page configuration and session state variables on page load.

    This function configures the layout of the page and initializes session
    state variables when the page is loaded.

    :return: None
    """
    st.set_page_config(layout="wide", initial_sidebar_state="expanded")
    set_all_session_state_variables()


def center_text(text: str) -> st.columns:
    """
    Centers a given text in a Streamlit application using a dynamic column layout.

    This function calculates the space required to center the text based on its length.
    It uses a simple linear formula to determine the column spacing and returns the
    appropriate column for text placement. If the calculated space is less than or
    equal to zero, a minimum spacing of 1 is used.

    :param text: The text to be centered.
    :type text: str
    :return: A Streamlit column for centered text placement.
    :rtype: st.columns
    """
    column_space = -0.02 * len(text) + 2.08
    if column_space <= 0:
        column_space = 1
    return st.columns([1, column_space, 2, 1, 1])[2]


def center_button() -> None:
    """
    Center-align buttons in the Streamlit app.

    This function applies CSS styles to center-align buttons within the
    Streamlit app.
    """
    st.markdown(
        """<style>
            div.stButton {
                display: flex;
                justify-content: center;
            }
            </style>""",
        unsafe_allow_html=True,
    )


def center_download_button() -> None:
    """
    Center-align download buttons in the Streamlit app.

    This function applies CSS styles to center-align download buttons within
    the Streamlit app.
    """
    st.markdown(
        """<style>
            div.stDownloadButton {
                display: flex;
                justify-content: center;
            }
            </style>""",
        unsafe_allow_html=True,
    )


def center_image() -> None:
    """
    Center-align images in the Streamlit app.

    This function applies CSS styles to center-align images within the
    Streamlit app.
    """
    st.markdown(
        """<style>
            .e115fcil2 {
                display: flex;
                justify-content: center;
            }
            </style>""",
        unsafe_allow_html=True,
    )
