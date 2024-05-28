import streamlit as st

import gui.st_utils as utils

utils.load_values()


def create_random_forest():
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            label_col, input_col = st.columns([2, 1])
            with label_col:
                st.button("N_Estimators")
                st.button("max_depth")
                st.button("min_samples_leaf")
                st.button("max_features")
                st.button("min_impurity_decrease")
                st.button("random_state")
                st.button("warm_start")
            with input_col:
                utils.int_input("n_estimators", default=35)
                utils.int_input("max_depth", default=6)
                utils.int_input("min_samples_leaf", default=2)
                utils.selectbox("max_features", ["log2","sqrt",1,2,3])
                utils.float_input("min_impurity_decrease", default=0.0)
                utils.selectbox("random_state", ["True", "False"])
                utils.bool_selectbox("warm_start")
        with col2:
            label_col, input_col = st.columns([2, 1])
            with label_col:
                st.button("max_samples")
                st.button("Criterion")
                st.button("min_samples_split")
                st.button("min_weight_fraction_leaf")
                st.button("max_leaf_nodes")
                st.button("bootstrap")
                st.button("ccp_alpha")
            with input_col:
                utils.selectbox("max_samples", options=[None,0.1,0.2,0.5,0.8,1.0], index=0)
                utils.selectbox("criterion", ["squarred_error", "absolute_error", "friedman_mse", "poisson"])
                utils.int_input("min_samples_split", default=2)
                utils.float_input("min_weight_fraction_leaf", 0.0)
                utils.selectbox("max_leaf_nodes", [None,1,2,3,4,5,10,15,20], index=0)
                utils.bool_selectbox("bootstrap")
                utils.float_input("ccp_alpha", default=0.0)


def get_random_forest_parameters():
    return {
        "n_estimators": st.session_state.n_estimators,
        "criterion": st.session_state.criterion,
        "max_depth": st.session_state.max_depth,
        "min_samples_split": st.session_state.min_samples_split,
        "min_samples_leaf": st.session_state.min_samples_leaf,
        "min_weight_fraction_leaf": st.session_state.min_weight_fraction_leaf,
        "max_features": st.session_state.max_features,
        "max_leaf_nodes": st.session_state.max_leaf_nodes,
        "min_impurity_decrease": st.session_state.min_impurity_decrease,
        "bootstrap": st.session_state.bootstrap,
        "random_state": st.session_state.random_state,
        "warm_start": st.session_state.warm_start,
        "ccp_alpha": st.session_state.ccp_alpha,
        "max_samples": st.session_state.max_samples,
    }
