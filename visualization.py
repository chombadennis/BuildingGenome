import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np
import datetime
import streamlit as st

# Aggregate Visulizations of the Clusters:
def timestampcombine(date,time):
    pydatetime = datetime.combine(date, time)
    return pydatetime

def ClusterUnstacker(df):
    df = df.unstack().reset_index() # code 1
    df['timestampstring'] = pd.to_datetime(df.Date.astype("str") + " " + df.level_2.astype("str")) # code 2
    #pd.to_datetime(df.Date  df.level_2) #map(timestampcombine, )
    df = df.dropna()
    return df

def DayvsClusterMaker(df):
    df.index = df.timestampstring
    df['Weekday'] = df.index.map(lambda t: t.date().weekday())
    df['Date'] = df.index.map(lambda t: t.date())
    df['Time'] = df.index.map(lambda t: t.time())
    # Convert 'Date' column to datetime objects before resampling
    df['Date'] = pd.to_datetime(df['Date'])
    DayVsCluster = df.resample('D').mean(numeric_only=True).reset_index(drop=True)
    DayVsCluster = pd.pivot_table(DayVsCluster, values=0, index='ClusterValue', columns='Weekday', aggfunc='count')
    DayVsCluster.columns = ['Mon','Tue','Wed','Thur','Fri','Sat','Sun']
    return DayVsCluster.T

def visualization(dfcluster_merged):
    if dfcluster_merged is not None:
        st.header("Visualizations of Data to Reveal Patterns of Behaviour over a Time Period")
        st.write(f"Setting Multilevel Indexing for Visualization. The dataframe below shows hierarchical division of the dtaframe- depending on the cluster values assigned by the K means algorithm:")
        dfcluster_merged_viz = dfcluster_merged.set_index(['ClusterValue', 'Date']).T.sort_index() # sets the dataframe with multilevel indexing, 'ClusterValue' as level 0
        st.write(f"Dataframe with multilevel indexing, 'ClusterValue' as level 0:")
        st.write(dfcluster_merged_viz)

        st.header(f"Visualization:")
        st.write(f"Extracted list of unique cluster numbers:")
        clusterlist = list(dfcluster_merged_viz.columns.get_level_values(0).unique())
        st.write(clusterlist)

        st.subheader(f'Plot of the daily energy consumption profiles grouped by clusters:')
        st.write(f"To visualize the cluster patterns, we first look at all the profiles at once grouped by cluster. We  iterate over each cluster to plot the daily energy consumption profiles. The x-axis represents the time of day, and the y-axis represents the total daily profile:")
        
        matplotlib.rcParams['figure.figsize'] = 20, 7
        styles2 = ['LightSkyBlue', 'b','LightGreen', 'g','LightCoral','r','SandyBrown','Orange','Plum','Purple','Gold','b']
        fig, ax = plt.subplots() # Creates a figure and an axes object for plotting
        for col, style in zip(clusterlist, styles2): # Iterates over each cluster number and its corresponding style.
            dfcluster_merged_viz[col].plot(ax=ax, legend=False, style=style, alpha=0.1, xticks=np.arange(0, 86400, 10800)) # Plots the daily profiles for the current cluster number with the specified style.

        ax.set_ylabel('Total Daily Profile')
        ax.set_xlabel('Time of Day')
        # Display the plot in the Streamlit app
        st.pyplot(fig)

        # Unstacking the Dataframe
        st.write(f"Having observed how the energy consumption has been clustered, there is no need for the hierarchy. The multilevel indexing is hence removed from dataframe:")
        dfclusterunstacked = ClusterUnstacker(dfcluster_merged_viz)
        st.write(dfclusterunstacked)
        st.write(f'The following pivot table is created from the unstacked dataframe above:')
        dfclusterunstackedpivoted = pd.pivot_table(dfclusterunstacked, values=0, index='timestampstring', columns='ClusterValue')
        st.write(dfclusterunstackedpivoted)

        st.subheader(f"Total Daily Load Profile Patterns")
        st.write(f"The plot below shows how the various load profiles have been clustered across the year to identify behaviour patterns of total Energy consumption per cluster through time.")
        fig2, ax2 = plt.subplots()
        clusteravgplot = dfclusterunstackedpivoted.resample('D').sum().replace(0, np.nan).plot(ax=ax2, style="^",markersize=15)
        clusteravgplot.set_ylabel('Daily Totals kWh')
        clusteravgplot.set_xlabel('Date')
        # Display the plot in the Streamlit app
        st.pyplot(fig2)

        st.subheader(f"Average Daily Load Profile Patterns")
        st.write(f"The plot below displays the average Energy Consumption in a 24 hour period per cluster.")
        fig3, ax3 = plt.subplots()
        dfclusterunstackedpivoted['Time'] = dfclusterunstackedpivoted.index.map(lambda t: t.time())
        dailyprofile = dfclusterunstackedpivoted.groupby('Time').mean().plot(ax = ax3, figsize=(20,7),linewidth=3, xticks=np.arange(0, 86400, 10800))
        dailyprofile.set_ylabel('Average Daily Profile kWh')
        dailyprofile.set_xlabel('Time of Day')
        dailyprofile.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
        # Display the plot in the Streamlit app
        st.pyplot(fig3)

        st.subheader(f"Plot to show the Clusters Observed (Load) per day")
        st.write(f"What can we deduce from the clusters observed  on average for every day of the Week")
        fig4, ax4 = plt.subplots()
        DayVsCluster = DayvsClusterMaker(dfclusterunstacked)
        DayVsClusterplot1 = DayVsCluster.plot(ax = ax4, figsize=(20,7),kind='bar',stacked=True)
        DayVsClusterplot1.set_ylabel('Number of Days in Each Cluster')
        DayVsClusterplot1.set_xlabel('Day of the Week')
        DayVsClusterplot1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cluster')
        # Display the plot in the Streamlit app
        st.pyplot(fig4)

        st.subheader(f"5th Plot shows the how much capacity each cluster (of Energy load) occupies per day")
        st.write(f"The plot below puts more perspective to the plot #4 above.")
        fig5, ax5 = plt.subplots()
        DayVsClusterplot2 = DayVsCluster.T.plot(ax = ax5, figsize=(20,7),kind='bar',stacked=True, color=['b','g','r','c','m','y','k']) #, color=colors2
        DayVsClusterplot2.set_ylabel('Number of Days in Each Cluster')
        DayVsClusterplot2.set_xlabel('Cluster Number')
        DayVsClusterplot2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # Display the plot in the Streamlit app
        st.pyplot(fig5) 

    