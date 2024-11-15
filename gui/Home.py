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
    utils.write_center("Intelligent planning of care staff")
    utils.write_center(
        "Welcome to our AI prediction tool for hospital occupancy. "
        "With advanced algorithms, we provide accurate forecasts to "
        "help streamline resource management and improve patient care.",
        tag="p",
    )

    add_vertical_space()

    st.caption(
        "Students: Aaron Bittel, Florian Paul, Adrian Lebmeier, Van Phuc Nguyen, "
        "Farhan Riftantya, Alexander Valerian"
    )
    st.caption(
        "Supervision: Prof. Dr.-Ing. Alexandra Teynor (THA), Julian Schanz (THA), "
        "Emily Schiller (XITASO), Felix Reichel (XITASO)"
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
