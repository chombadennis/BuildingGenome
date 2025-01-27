import streamlit as st
from predict_page import show_predict_page
from explore_page import show_explore_page

st.title('MACHINE LEARNING PROJECT OF THE BUILDING GENOME PROJECT')

st.header('SECTION A: Clustering of the Daily Energy Consumption load profiles for the Building Data Genome Project')

st.write(
    """
    Here, we use the clustering algorithm to group buildings with similar energy consumption patterns and identify patterns
    or anomalies. Through clustering, we aim to enhance energy management, detect anomalies, and improve demand forecasting.
    """
)

# Sidebar for navigation
page = st.sidebar.selectbox("Select a Page", ("Predict", "Explore"))

if page == "Predict":
    show_predict_page()
elif page == "Explore":
    show_explore_page()