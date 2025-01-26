import streamlit as st
import pandas as pd

def display_cluster_info(df_pivot_w_clusters):
    st.title('Cluster Information')
    ok = st.button('Identify Specific Cluster')
    if ok:
        # Input fields for the user to specify the date and hour
        user_date = st.date_input('Enter a date to check cluster:', value=None, min_value=None, max_value=None, key='date_input')
        user_hour = st.number_input('Enter an hour to check cluster (0-23):', min_value=0, max_value=23, step=1, key='hour_input')

        if user_date and user_hour is not None:
            # Filter the DataFrame to get the predicted cluster number and energy values for the specified date and hour
            filtered_df = df_pivot_w_clusters[(df_pivot_w_clusters['Date'] == user_date) & (df_pivot_w_clusters['Time'] == user_hour)]
            if not filtered_df.empty:
                st.write(f"Predicted cluster number for {user_date} at {user_hour}:00 is {filtered_df['ClusterNo'].values[0]}")
                st.write(f"Energy value used: {filtered_df['energy_values'].values[0]}")
            else:
                st.write("No data available for the specified date and hour.")