import pickle
import streamlit as st
import pickle
import numpy as np
import datetime
import pandas as pd
from datetime import datetime
import os
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


def show_explore_page():
    st.title("Explore Cluster Information")

    # Check if the 'dfcluster_merged.pkl' file exists
    if "dfcluster_merged" not in st.session_state:
        # Try loading the pickle file if it's not in the session state yet
        try:
            if os.path.exists('dfcluster_merged.pkl'):
                # Load the DataFrame from the pickle file if it exists
                with open('dfcluster_merged.pkl', 'rb') as f:
                    st.session_state.dfcluster_merged = pickle.load(f)
                    st.session_state.clustering_done = True
                st.success("Clustering data has been loaded successfully.")
            else:
                st.error("Clustering data not found. Please run clustering first on the 'Predict' page.")
                return  # Don't proceed if the file doesn't exist
        except FileNotFoundError:
            st.error("Clustering data file not found! Please run clustering first.")
            return  # Don't proceed if there was an error loading the file

    # Ensure  and proceed if clustering has been done
    if "clustering_done" in st.session_state and st.session_state.clustering_done:
        # Retrieve the clustered DataFrame from session state
        dfcluster_merged = st.session_state.dfcluster_merged
        st.write("Clustered DataFrame (with 'ClusterValue'):")
        st.write(dfcluster_merged.head(50))

        # User inputs: date and time
        user_date = st.date_input("Enter a date to check cluster and energy load:")
        user_time = st.time_input("Enter a time (HH:MM:SS):")

        if user_date is not None and user_time is not None:
            # Convert user_time to a `datetime.time` object
            time_obj = user_time

            if time_obj in dfcluster_merged.columns:
                # Filter the DataFrame for the selected date
                filtered_df = dfcluster_merged[dfcluster_merged["Date"] == user_date]

                if not filtered_df.empty:
                    # Retrieve the cluster value and energy load
                    cluster_value = filtered_df.iloc[0]["ClusterValue"]
                    energy_load = filtered_df.iloc[0][time_obj]

                    st.write(f"Cluster for {user_date} at {time_obj} is **{cluster_value}**.")
                    st.write(f"Energy load for the specified time is **{energy_load} KWh**.")
                else:
                    st.warning("No data available for the specified date.")
            else:
                st.error(f"No energy values found for the time '{user_time}'. Please enter a valid time.")
    else:
        st.error("Clustered data unavailable. Please complete clustering on the 'Predict' page first.")




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

        # Save the processed dataframe to a pickle file to persist it
        with open('dfcluster_merged.pkl', 'wb') as f:
            pickle.dump(dfcluster_merged, f)

        #store in session state
        st.session_state.dfcluster_merged = dfcluster_merged
        st.session_state.clustering_done = True

        # Call the visualization function
        visualization(dfcluster_merged)

    else:
        st.error("The DataFrame is empty or not available.")
    

