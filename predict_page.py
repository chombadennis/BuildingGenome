import streamlit as st
import pickle
import numpy as np
import datetime
from explore_page import load_model, normalize_dataframe, reorder_clusters, show_explore_page
import pandas as pd

import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

from scipy.cluster.vq import kmeans, vq, whiten
from scipy.spatial.distance import cdist
import numpy as np
from datetime import datetime

cl_centers = load_model()


def show_predict_page(): 
    st.header('K Means Clustering of the Daily Load Profiles')

    st.write("""### Upload file:""")

    # Allow the user to upload a CSV file
    file = st.file_uploader("Choose a CSV file", type="csv", key='100k')
    if file is not None:
        try:
            # Read the uploaded file into a DataFrame
            df = pd.read_csv(file)
            st.write("Uploaded DataFrame:")
            st.write(df)

            # Rename the columns
            df.rename(columns={df.columns[0]: 'timestamp', df.columns[1]: 'energy_values'}, inplace=True)
            st.write("DataFrame with renamed columns:")
            st.write(df)

            # Ensure the DataFrame has the correct structure
            if 'timestamp' in df.columns and 'energy_values' in df.columns:
                # Normalize the DataFrame
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df_norm = normalize_dataframe(df)
                st.write("Normalized DataFrame:")
                st.write(df_norm)

                # Pivot the DataFrames
                df_pivot = pd.pivot_table(df, values='energy_values', index='Date', columns='Time', aggfunc='mean')
                df_pivot_norm = pd.pivot_table(df_norm, values='energy_values', index='Date', columns='Time', aggfunc='mean')
                st.write("Pivot dataframe of normalized Data:")
                st.write(df_pivot_norm)

                # Ensure the pivoted DataFrame has 24 columns
                if df_pivot_norm.shape[1] == 24:
                    # Convert the pivoted DataFrame to a numpy matrix
                    df_pivot_norm_matrix = np.matrix(df_pivot_norm.dropna())

                    # Use vq to assign clusters to the new data
                    cluster_indices, _ = vq(df_pivot_norm_matrix, cl_centers)

                    # Create a dataframe of the cluster_indices:
                    clusterdf = pd.DataFrame(cluster_indices, columns=['ClusterNo'])
                    # Concatenate with df_pivot:
                    df_pivot_w_clusters = pd.concat([df_pivot.dropna().reset_index(), clusterdf], axis=1)
                    st.write("Below table shows the original dataframe with the unnormalized data. It has been concatenated with the dataframe containing the assigned cluster number. These clusters were got from the k-means clustering model.")
                    st.write(df_pivot_w_clusters)

                    # Call the reorder_clusters function
                    dfcluster_merged = reorder_clusters(df_pivot_w_clusters)                 

                    if isinstance(dfcluster_merged, pd.DataFrame):
                        st.write("Download dataframe that has the required clusters here. You can identify cluster number and energy load of a specific time on a specific day by uploading this csv file in the 'explore'page")
                    else:
                        st.wsrite("dfcluster_merged NOT a DataFrame.")

                    # Download button section starts here
                    csv = dfcluster_merged.to_csv(index=True)  # Convert dataframe to CSV (with index)
                    st.download_button(label="Download Clustered Data as CSV", data=csv, file_name="clustered_data.csv", mime="text/csv")
                    # Download button section ends here
        
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
    else:
        st.error("Please upload the CSV file to proceed.")


