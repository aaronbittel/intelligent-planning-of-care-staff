import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.add_vertical_space import add_vertical_space
from streamlit_extras.metric_cards import style_metric_cards
import gui.st_utils as utils
import plotly.express as px

########################################################################################
#   SETTING VARIABLES                                                                  #
########################################################################################

utils.on_page_load()
utils.load_values()


iframe = utils.create_iframe_link()


########################################################################################
#   METHODS                                                                            #
########################################################################################


def set_metric(model: str, metric: str) -> None:
    """
    Sets and displays a metric value for a given model in Streamlit.

    This function retrieves the specified metric value for the given model from the
    session state, rounds it to two decimal places, and displays it using Streamlit's
    `st.metric` function.

    :param model: The name of the model.
    :type model: str
    :param metric: The name of the metric to be displayed.
    :type metric: str
    """
    if st.session_state.metrics[model].get(metric, None):
        st.metric(
            f"{model} {metric}",
            value=round(st.session_state.metrics[model][metric], 2),
        )


def get_best_performing_modelname() -> str:
    """
    Retrieves the name of the best performing model based on the lowest RMSE value.

    This function iterates through the metrics stored in `st.session_state` and
    identifies the model with the lowest RMSE. The model name is formatted to replace
    hyphens with underscores and converted to lowercase.

    :return: The name of the best performing model with the lowest RMSE.
    :rtype: str
    """
    best_rmse_model = None
    lowest_rmse = float("inf")
    for model, metric in st.session_state.metrics.items():
        if metric["RMSE"] < lowest_rmse:
            lowest_rmse = metric["RMSE"]
            best_rmse_model = model

    return best_rmse_model.replace("-", "_").lower()


def create_weekly_figure() -> px.bar:
    """
    Creates a weekly occupancy figure using Plotly.

    This function generates a bar chart representing the average occupancy per weekday.
    The data is extracted from the DataFrame stored in `st.session_state.df`, which is
    grouped by weekdays.

    :return: The generated bar chart showing average occupancy per weekday.
    :rtype: px.Figure
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
    weekly_data.index = weekly_data.index.map(lambda x: weekday_names[x])

    fig_weekly = px.bar(
        weekly_data,
        y="occupancy",
        x=weekly_data.index,
        color=weekly_data.index,
        color_discrete_map={day: "#FF4B4B" for day in weekday_names},
        labels={"y": "Weekday", "x": "Occupancy"},
        orientation="v",
    )
    fig_weekly.update_layout(showlegend=False)

    fig_weekly.update_yaxes(range=[max(weekly_data) * 0.7, max(weekly_data) * 1.1])

    return fig_weekly


def convert_df() -> bytes:
    """
    Converts the DataFrame to a CSV file and encodes it in UTF-8.

    This function reads a CSV file from the 'output' directory based on the filename of
    the model with the best RMSE, and converts it to a CSV file encoded in UTF-8.

    :return: The CSV file encoded in UTF-8.
    :rtype: bytes
    """
    df = pd.read_csv(os.path.join("output", f"latest_{filename_best_rmse}.csv"))
    return df.to_csv().encode("utf-8")


def show_download_button() -> bool:
    """
    Determines whether the download button should be shown.

    This function checks if the selected type in the session state is 'forecast' and if
    any of the models in the session state metrics have a defined RMSE value. If both
    conditions are met, it returns True, indicating that the download button should be
    shown.

    :return: True if the download button should be shown, otherwise False.
    :rtype: bool
    """
    return st.session_state.selected_type == utils.SelectedType.FORECAST and any(
        metric["RMSE"] for metric in st.session_state.metrics.values()
    )


########################################################################################
#   PAGE STRUCTURE                                                                     #
########################################################################################


page_link_container = st.container()

page_selector_container = st.container()
_, forecast_btn_col, select_weekly_col, _ = page_selector_container.columns(
    [3, 1, 1, 3]
)

st.divider()

if st.session_state.selected_view == utils.SelectedView.FORECAST_VIEW:
    forecast_text_container = st.container()
    grafana_container = st.container()

    add_vertical_space(2)

    rmse_container = st.container()
    s_rmse_col, rf_rmse_col, hw_rmse_col = rmse_container.columns(3)
    mape_container = st.container()
    s_mape_col, rf_mape_col, hw_mape_col = mape_container.columns(3)
    mae_container = st.container()
    s_mae_col, rf_mae_col, hw_mae_col = mae_container.columns(3)

    add_vertical_space(2)

    if st.session_state.selected_type == utils.SelectedType.FORECAST:
        download_btn_container = st.container()

if st.session_state.selected_view == utils.SelectedView.WEEKLY_VIEW:
    left_col, _ = st.columns(2)
    weekly_graph_container = left_col.container(border=True)


########################################################################################
#   CREATING WIDGETS                                                                   #
########################################################################################


with page_link_container:
    st.page_link(page="pages/1_Setup.py", label="← back to setup")


with page_selector_container:
    with forecast_btn_col:
        select_forecast_btn = st.button(
            "Forecast",
            type="primary"
            if st.session_state.selected_view == utils.SelectedView.FORECAST_VIEW
            else "secondary",
            use_container_width=True,
        )
    with select_weekly_col:
        select_weekly_btn = st.button(
            "Weekly",
            type="primary"
            if st.session_state.selected_view == utils.SelectedView.WEEKLY_VIEW
            else "secondary",
            use_container_width=True,
        )


if st.session_state.selected_view == utils.SelectedView.FORECAST_VIEW:
    with forecast_text_container:
        st.title("Forecast")
        st.write("Interact with the graph to take a detailed look at the predictions.")

    with grafana_container:
        components.iframe(src=iframe, height=1000)

    if show_download_button():
        filename_best_rmse = get_best_performing_modelname()

        csv_bytes = convert_df()
        with download_btn_container:
            download_btn = st.download_button(
                label="Download CSV",
                data=csv_bytes,
                file_name="prediction.csv",
                mime="text/csv",
                type="primary",
            )


if st.session_state.selected_view == utils.SelectedView.WEEKLY_VIEW:
    with weekly_graph_container:
        st.write(
            "<h3 style='text-align:center'>Average Occupancy per Weekday</h3>",
            unsafe_allow_html=True,
        )

        fig_weekly = create_weekly_figure()
        st.plotly_chart(fig_weekly, use_container_width=False)


########################################################################################
#   FUNCTIONALITY                                                                      #
########################################################################################


if st.session_state.selected_view == utils.SelectedView.FORECAST_VIEW:
    with rmse_container.container():
        with s_rmse_col:
            set_metric("Sarima", "RMSE")
        with rf_rmse_col:
            set_metric("Random-Forest", "RMSE")
        with hw_rmse_col:
            set_metric("Holt-Winters", "RMSE")

    with mape_container.container():
        with s_mape_col:
            set_metric("Sarima", "MAPE")
        with rf_mape_col:
            set_metric("Random-Forest", "MAPE")
        with hw_mape_col:
            set_metric("Holt-Winters", "MAPE")

    with mae_container.container():
        with s_mae_col:
            set_metric("Sarima", "MAE")
        with rf_mae_col:
            set_metric("Random-Forest", "MAE")
        with hw_mae_col:
            set_metric("Holt-Winters", "MAE")


if select_forecast_btn:
    st.session_state.selected_view = utils.SelectedView.FORECAST_VIEW
    st.rerun()

if select_weekly_btn:
    st.session_state.selected_view = utils.SelectedView.WEEKLY_VIEW
    st.rerun()


########################################################################################
#   STYLING                                                                            #
########################################################################################


utils.center_download_button()
style_metric_cards()