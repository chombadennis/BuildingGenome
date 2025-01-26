import pickle
import streamlit as st
import pickle
import numpy as np
import datetime
import pandas as pd
from datetime import datetime
from visualization import visualization

def load_model():
    with open('kmeans_centers.pkl', 'rb') as file:
        centers = pickle.load(file)
    return centers

def normalize_dataframe(df):
    # Extract time and date components before normalization
    df['Time'] = df['timestamp'].map(lambda t: t.time())
    df['Date'] = df['timestamp'].map(lambda t: t.date())

    # Select only numerical columns for normalization
    numeric_cols = df.select_dtypes(include=['number']).columns
    df_norm = (df[numeric_cols] - df[numeric_cols].mean()) / (df[numeric_cols].max() - df[numeric_cols].min())

    # Add back the time and date columns to the normalized dataframe
    df_norm['Time'] = df['Time']
    df_norm['Date'] = df['Date']
    
    return df_norm

import streamlit as st
import pandas as pd

def show_explore_page():
    st.title('Explore Cluster Information')

    if 'df_pivot_w_clusters' in st.session_state:
        df_pivot_w_clusters = st.session_state.df_pivot_w_clusters

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
    else:
        st.write("Please run the prediction first to generate cluster information.")

def reorder_clusters(df_pivot_w_clusters):
    if df_pivot_w_clusters is not None:
        st.header('Reordering Clusters:')
        st.write(f'Next, the clustering numbers are reordered such that the clusters that have values representing highest energy consumption are assigned the highest cluster numbers:')

        # Calculating the total average consumption for each cluster.
        x = df_pivot_w_clusters.groupby('ClusterNo').mean(numeric_only=True).sum(axis=1).sort_values()
        # Create a DataFrame from the sorted result and assign new cluster numbers.
        st.write(f"The total average consumption of each cluster:")
        st.write(x)

        x_new = pd.DataFrame(x.reset_index()) # Creates a DataFrame from the sorted result
        st.write(f" Dataframe of the sorted result:")
        st.write(x_new)
        x_new['ClusterValue'] = x_new.index # Creates a new column 'ClusterValue' and assigns new cluster numbers (0, 1, 2, ...) based on the sorted order.
        x_new = x_new.set_index('ClusterNo') # Sets 'ClusterNo' as the index.
        x_new = x_new.drop([0], axis=1) # Drops the unnecessary column with sum of averages of the clusters
        st.write(f'A bit confusing right?')
        st.write(f"Reassigning the cluster numbers with the order 0,1,2,3. =Such that that 0 is the lowest consuming cluster, followed by 1, 2, then 3 is the highest consuming cluster. Dataframe showing what value the original cluster was reassigned to:")
        st.write(x_new)
        # Merge the new cluster numbers into the original DataFrame.
        dfcluster_merged = df_pivot_w_clusters.merge(x_new, how='outer', left_on='ClusterNo', right_index=True)
        # dailyclusters.merge(...): Merges the original dailyclusters DataFrame with the DataFrame x containing the new cluster numbers.
        # how='outer': Performs a full outer join, including all rows from both DataFrames.
        # left_on='ClusterNo': Specifies the column to join on from the left DataFrame (dailyclusters).
        # right_index=True: Specifies that the index of the right DataFrame (x) should be used as the join key.
        dfcluster_merged = dfcluster_merged.drop(['ClusterNo'], axis=1) # drops the 'ClusterNo' column because it is now unnecessary since we reordered the clusters to 'ClusterNo2' order
        st.write(f"The dataframe below shows the original but pivoted dataframe conataining unnormalized energy consumption data with the reordered and reassigned cluster numbers in the last column:")
        st.write(dfcluster_merged)

        # Call the visualization function
        visualization(dfcluster_merged)
    else:
        st.error("The DataFrame is empty or not available.")
    


