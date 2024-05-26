import os

import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space

import gui.st_utils as utils

########################################################################################
#   SETTING VARIABLES                                                                  #
########################################################################################


utils.on_page_load()


########################################################################################
#   PAGE STRUCTURE                                                                     #
########################################################################################


logo_container = st.container()
main_container = st.container(border=True)
main_text_container = main_container.container()

add_vertical_space()

start_btn_container = st.container()


########################################################################################
#   CREATING WIDGETS                                                                   #
########################################################################################


with logo_container:
    logo = st.image(image=os.path.join("gui", "logo.jpg"), width=150)


with main_text_container:
    st.write("<h1 style='text-align:center'>Home</h1>", unsafe_allow_html=True)
    st.write(
        "<p style='text-align:center'>"
        "Welcome to our AI prediction tool for hospital occupancy. "
        "With advanced algorithms, we provide accurate forecasts to "
        "help streamline resource management and improve patient care."
        "</p>",
        unsafe_allow_html=True,
    )


with start_btn_container:
    start_btn = st.button("START", type="primary")


########################################################################################
#   FUNCTIONALITY                                                                      #
########################################################################################


if start_btn:
    st.switch_page("pages/1_Setup.py")


########################################################################################
#   STYLING                                                                            #
########################################################################################


utils.center_image()
utils.center_button()
