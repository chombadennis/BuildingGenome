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
    
    # Allow the user to upload a CSV file
    file_cl = st.file_uploader("Choose a CSV file", type="csv", key='low')
    
    # st.subheader('Click below to get energy load and cluster value information')
    if file_cl is not None:
        try:
            # Read the uploaded file into a DataFrame
            dfcluster_merged = pd.read_csv(file_cl)
            st.write("This is the uploaded file with the cluster values:")
            
            # Ensure the 'Date' column exists
            if 'Date' not in dfcluster_merged.columns:
                st.error("The uploaded file must contain a 'Date' column.")
                return
            
            # Process the 'Date' column
            #dfcluster_merged = dfcluster_merged.iloc[:, 1:]  # Drop the first column
            dfcluster_merged['Date'] = pd.to_datetime(dfcluster_merged['Date']).dt.date  # Ensure date format (YYYY-MM-DD)
            dfcluster_merged.set_index('Date', inplace=True)  # Set 'Date' as the index

            st.write(dfcluster_merged)  # Display a preview of the DataFrame

            # User inputs: date and time
            user_date = st.date_input("Enter a date in 2015 to check cluster and energy load:", min_value=pd.to_datetime('2010-01-01').date())
            user_time = st.time_input("Enter a time to the nearest hour (HH:MM:SS):")

            if user_date and user_time:
                # Convert user_time to string matching the column format
                time_str = user_time.strftime("%H:%M:%S")

                if time_str in dfcluster_merged.columns:
                    # Filter the DataFrame for the selected date
                    filtered_df = dfcluster_merged.loc[dfcluster_merged.index == user_date]

                    if not filtered_df.empty:
                        # Retrieve the cluster value and energy load
                        cluster_value = filtered_df.get("ClusterValue", [None])[0]
                        energy_load = filtered_df.get(time_str, [None])[0]

                        if cluster_value is not None and energy_load is not None:
                            st.write(f"Cluster for {user_date} at {time_str} is **{cluster_value}**.")
                            st.write(f"Energy load for the specified time is **{energy_load} KWh**.")
                        else:
                            st.warning("Required information could not be found in the DataFrame.")
                    else:
                        st.warning("No data available for the specified date.")
                else:
                    st.error(f"No energy values found for the time '{time_str}'. Please enter a valid time.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload the file from the previous page.")






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
        st.write(f"Dataframe showing total energy cosumption per cluster sorted in ascending order:")
        st.write(x_new)
        x_new['ClusterValue'] = x_new.index # Creates a new column 'ClusterValue' and assigns new cluster numbers (0, 1, 2, ...) based on the sorted order.
        x_new = x_new.set_index('ClusterNo') # Sets 'ClusterNo' as the index.
        x_new = x_new.drop([0], axis=1) # Drops the unnecessary column with sum of averages of the clusters
        st.write(f'The way cluster values are assigned is confusing right?')
        st.write(f"Below, cluster values have been reassigned in the order 0,1,2,3. Such that that 0 is the cluster with the lowest energy load, followed by 1, 2, then 3 as the highest consuming cluster. The dataframe below shows what value the original cluster number was reassigned to:")
        st.write(x_new)
        # Merge the new cluster numbers into the original DataFrame.
        dfcluster_merged = df_pivot_w_clusters.merge(x_new, how='outer', left_on='ClusterNo', right_index=True)
        # dailyclusters.merge(...): Merges the original dailyclusters DataFrame with the DataFrame x containing the new cluster numbers.
        # how='outer': Performs a full outer join, including all rows from both DataFrames.
        # left_on='ClusterNo': Specifies the column to join on from the left DataFrame (dailyclusters).
        # right_index=True: Specifies that the index of the right DataFrame (x) should be used as the join key.
        dfcluster_merged = dfcluster_merged.drop(['ClusterNo'], axis=1) # drops the 'ClusterNo' column because it is now unnecessary since we reordered the clusters to 'ClusterNo2' order
        st.write(f"The dataframe below shows the original but pivoted dataframe containing unnormalized energy consumption data with the reordered and reassigned cluster numbers in the last column. Download it to view clusters.")
        st.subheader(f"DataFrame with the required Clusters")
        st.write(dfcluster_merged)

        #store in session state
        st.session_state.dfcluster_merged = dfcluster_merged
        st.session_state.clustering_done = True

        # Call the visualization function
        visualization(dfcluster_merged)

        return(dfcluster_merged)

    else:
        st.error("The DataFrame is empty or not available.")
    

